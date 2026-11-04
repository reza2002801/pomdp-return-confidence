"""
Continuous-time physical models for the LQG benchmark.

Each builder returns (Ac, Bc, meta). Discretization and noise handling live in plant.py.
Illustrative parameters are documented in the JSON configs; they are not claimed as a
standard public benchmark unless a source is cited there.
"""

import numpy as np


def continuous_model(cfg):
    kind = cfg["system"]
    builders = {
        "mass_spring_damper": mass_spring_damper,
        "coupled_masses": coupled_masses,
        "quarter_car": quarter_car,
        "two_room_thermal": two_room_thermal,
        "drone_hover": drone_hover,
        "cartpole_linear": cartpole_linear,
    }
    if kind not in builders:
        raise ValueError(f"unknown system {kind}")
    return builders[kind](cfg["physics"])


def mass_spring_damper(p):
    """
    m xdd + b xd + k x = u.
    State (x, xd), input force u [N].
    """
    m, b, k = p["m"], p["b"], p["k"]
    Ac = np.array([[0.0, 1.0], [-k / m, -b / m]])
    Bc = np.array([[0.0], [1.0 / m]])
    meta = {
        "state_labels": [r"$x$", r"$\dot x$"],
        "state_units": ["m", "m/s"],
        "input_labels": [r"$u$"],
        "input_units": ["N"],
        "notes": [
            "Mass-spring-damper: m xdd + b xd + k x = u.",
            f"m={m} kg, b={b} N s/m, k={k} N/m.",
        ],
    }
    return Ac, Bc, meta


def coupled_masses(p):
    """
    Two masses with grounded springs/dampers and a coupling spring/damper.
    Force u acts on mass 1.

    m1 x1dd = -k1 x1 - b1 x1d - kc (x1 - x2) - bc (x1d - x2d) + u
    m2 x2dd = -k2 x2 - b2 x2d - kc (x2 - x1) - bc (x2d - x1d)

    Coupling signs: each mass feels a restoring force toward the other.
    """
    m1, m2 = p["m1"], p["m2"]
    k1, k2, kc = p["k1"], p["k2"], p["kc"]
    b1, b2, bc = p["b1"], p["b2"], p["bc"]
    # accelerations
    # x1dd = (-(k1+kc)/m1) x1 + (-(b1+bc)/m1) x1d + (kc/m1) x2 + (bc/m1) x2d + u/m1
    # x2dd = (kc/m2) x1 + (bc/m2) x1d + (-(k2+kc)/m2) x2 + (-(b2+bc)/m2) x2d
    Ac = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(k1 + kc) / m1, -(b1 + bc) / m1, kc / m1, bc / m1],
        [0.0, 0.0, 0.0, 1.0],
        [kc / m2, bc / m2, -(k2 + kc) / m2, -(b2 + bc) / m2],
    ])
    Bc = np.array([[0.0], [1.0 / m1], [0.0], [0.0]])
    # verify coupling: force on 1 from relative displacement (x1-x2) is -kc(x1-x2)
    # so x1dd gets -kc/m1 x1 + kc/m1 x2; x2dd gets +kc/m2 x1 - kc/m2 x2.
    assert Ac[1, 0] < 0 and Ac[1, 2] > 0 and Ac[3, 0] > 0 and Ac[3, 2] < 0
    meta = {
        "state_labels": [r"$x_1$", r"$\dot x_1$", r"$x_2$", r"$\dot x_2$"],
        "state_units": ["m", "m/s", "m", "m/s"],
        "input_labels": [r"$u$"],
        "input_units": ["N"],
        "notes": [
            "Coupled masses with grounded and coupling spring/damper. Control force on mass 1.",
            "Coupling: m1 sees -kc(x1-x2)-bc(x1d-x2d); m2 sees -kc(x2-x1)-bc(x2d-x1d).",
        ],
    }
    return Ac, Bc, meta


def quarter_car(p):
    """
    Active quarter-car. States are displacements from static equilibrium and velocities:
      S = (zs, zs_dot, zu, zu_dot)
    where zs is sprung (body) displacement and zu is unsprung (wheel) displacement.

    Force balance (road displacement zr enters through the tire spring only):
      ms zsdd = -ks (zs - zu) - bs (zs_dot - zu_dot) + u
      mu zudd = +ks (zs - zu) + bs (zs_dot - zu_dot) - kt (zu - zr) - u

    The active force u acts equally and oppositely between body and wheel.

    Road disturbance: we treat zr as filtered continuous-time white noise through a
    disturbance channel G_c that injects into the tire-spring term. Specifically the
    continuous process noise channel multiplies a scalar white intensity into the
    acceleration of the unsprung mass as (kt/mu) * w_road, which is the linearization
    of + (kt/mu) zr. Gaussian white noise is not a differentiable road-displacement
    signal; it is a formal disturbance that produces a discrete covariance via Van Loan.
    """
    ms, mu = p["ms"], p["mu"]
    ks, bs, kt = p["ks"], p["bs"], p["kt"]
    Ac = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-ks / ms, -bs / ms, ks / ms, bs / ms],
        [0.0, 0.0, 0.0, 1.0],
        [ks / mu, bs / mu, -(ks + kt) / mu, -bs / mu],
    ])
    Bc = np.array([[0.0], [1.0 / ms], [0.0], [-1.0 / mu]])
    # road channel: continuous G such that G w contributes (kt/mu) w to wheel acceleration
    Gc = np.array([[0.0], [0.0], [0.0], [kt / mu]])
    meta = {
        "state_labels": [r"$z_s$", r"$\dot z_s$", r"$z_u$", r"$\dot z_u$"],
        "state_units": ["m", "m/s", "m", "m/s"],
        "input_labels": [r"$u$"],
        "input_units": ["N"],
        "process_channel": Gc,
        "report_channels": {
            "body_displacement": 0,
            "suspension_deflection": "zs-zu",
            "wheel_displacement": 2,
        },
        "notes": [
            "Active quarter-car: body and wheel masses, suspension spring/damper, tire stiffness.",
            "Active force +u on the body and -u on the wheel.",
            "Road enters as a formal white disturbance through the tire spring channel; "
            "it is not a differentiable road-displacement trajectory.",
        ],
    }
    return Ac, Bc, meta


def two_room_thermal(p):
    """
    Two-room temperatures relative to a desired equilibrium T*.

    C1 T1_dot = -a1 (T1 - T_ext_dev) - a12 (T1 - T2) + u1
    C2 T2_dot = -a2 (T2 - T_ext_dev) - a12 (T2 - T1) + u2

    With deviation variables about an equilibrium where exterior deviation is zero:
      C1 t1_dot = -(a1 + a12) t1 + a12 t2 + u1
      C2 t2_dot = a12 t1 - (a2 + a12) t2 + u2

    Units: Ci in J/K (or W s / K), ai in W/K, ui in W (heating/cooling power deviation).
    Negative ui means cooling, or heating below the equilibrium level. The plant is not
    heating-only.
    """
    C1, C2 = p["C1"], p["C2"]
    a1, a2, a12 = p["a1"], p["a2"], p["a12"]
    Ac = np.array([
        [-(a1 + a12) / C1, a12 / C1],
        [a12 / C2, -(a2 + a12) / C2],
    ])
    Bc = np.array([[1.0 / C1, 0.0], [0.0, 1.0 / C2]])
    assert Ac[0, 0] < 0 and Ac[1, 1] < 0 and Ac[0, 1] > 0 and Ac[1, 0] > 0
    meta = {
        "state_labels": [r"$t_1$", r"$t_2$"],
        "state_units": ["K", "K"],
        "input_labels": [r"$u_1$", r"$u_2$"],
        "input_units": ["W", "W"],
        "notes": [
            "Two-room thermal deviations from a desired equilibrium.",
            "Ci in J/K, conductances ai in W/K, inputs in W (positive heats, negative cools).",
            "The inputs are unconstrained power deviations, not heating-only actuators.",
        ],
    }
    return Ac, Bc, meta


def drone_hover(p):
    """
    Planar small-angle hover-relative model.

    State S = (x, z, xd, zd, phi, phid):
      x horizontal, z vertical (altitude), phi pitch.

    Sign convention: positive phi tips the nose toward +x, so the thrust tilts and
    produces a positive horizontal acceleration when phi is positive (small-angle):
      xdd ≈ g phi
      zdd ≈ u_thrust / m      (thrust deviation above hover thrust mg)
      phidd = tau / Iyy

    Continuous linearization about hover (thrust = mg cancelled by gravity):
      Ac maps phi -> xdd by g, and Bc maps thrust deviation and pitch torque.
    Hover-relative inputs: u = (delta_thrust [N], torque [N m]). The control-cost term
    penalizes deviations from hover, not total electrical energy.
    """
    m, Iyy, g = p["m"], p["Iyy"], p["g"]
    Ac = np.zeros((6, 6))
    Ac[0, 2] = 1.0
    Ac[1, 3] = 1.0
    Ac[2, 4] = g          # xdd = g * phi
    Ac[4, 5] = 1.0
    Bc = np.zeros((6, 2))
    Bc[3, 0] = 1.0 / m    # zdd from thrust deviation
    Bc[5, 1] = 1.0 / Iyy  # phidd from torque
    meta = {
        "state_labels": [r"$x$", r"$z$", r"$\dot x$", r"$\dot z$", r"$\phi$", r"$\dot\phi$"],
        "state_units": ["m", "m", "m/s", "m/s", "rad", "rad/s"],
        "input_labels": [r"$\delta T$", r"$\tau$"],
        "input_units": ["N", "N·m"],
        "notes": [
            "Planar linearized hover: positive pitch toward +x gives xdd = g phi.",
            "Vertical acceleration from thrust deviation above hover thrust mg.",
            "Cost penalizes input deviations from hover, not total thrust energy.",
            "Guarantees apply to this linearization, not the nonlinear rigid-body drone.",
        ],
    }
    return Ac, Bc, meta


def cartpole_linear(p):
    """
    Linearized frictionless cart-pole about upright. Same convention as the existing
    cartpole_observer experiment: positive angle leans toward positive cart position.
    Retained as a continuous linear-Gaussian reference only.
    """
    M, m, ell, g = p["M"], p["m"], p["l"], p["g"]
    Ac = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -(m * g) / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (M + m) * g / (M * ell), 0.0],
    ])
    Bc = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * ell)]])
    meta = {
        "state_labels": [r"$x$", r"$\dot x$", r"$\theta$", r"$\dot\theta$"],
        "state_units": ["m", "m/s", "rad", "rad/s"],
        "input_labels": [r"$u$"],
        "input_units": ["N"],
        "notes": [
            "Linearized cart-pole about upright; guarantees do not cover the nonlinear plant.",
            "Same sign convention as cartpole_observer: positive theta leans toward +x.",
        ],
    }
    return Ac, Bc, meta
