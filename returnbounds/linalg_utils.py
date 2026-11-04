"""Shared linear-algebra helpers for discrete LQG plants."""

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, solve_discrete_are


def zoh(Ac, Bc, dt):
    n, m = Bc.shape
    block = np.zeros((n + m, n + m))
    block[:n, :n] = Ac
    block[:n, n:] = Bc
    E = expm(block * dt)
    return E[:n, :n], E[:n, n:]


def discrete_process_cov(Ac, G, Qw, dt):
    """
    Sigma_W = integral_0^dt exp(Ac tau) G Qw G' exp(Ac' tau) d tau
    via the Van Loan matrix exponential.
    """
    n = Ac.shape[0]
    Q = G @ Qw @ G.T
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -Ac
    M[:n, n:] = Q
    M[n:, n:] = Ac.T
    E = expm(M * dt)
    return E[n:, n:].T @ E[:n, n:]


def discrete_process_cov_from_intensity(Ac, intensity, dt):
    """Identity channel: G = I, Qw = intensity (matrix or scalar)."""
    n = Ac.shape[0]
    Qw = np.eye(n) * float(intensity) if np.ndim(intensity) == 0 else np.asarray(intensity, dtype=float)
    return discrete_process_cov(Ac, np.eye(n), Qw, dt)


def lqr_gain(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def kalman_gain(A, H, SW, SV):
    """
    Steady-state prediction covariance and measurement-update gain
    S_hat = Z + L (Y - H Z).
    """
    # Dual discrete Riccati for the prediction covariance of the process A, noise SW,
    # observation H, measurement noise SV.
    P_pred = solve_discrete_are(A.T, H.T, SW, SV)
    S = H @ P_pred @ H.T + SV
    L = P_pred @ H.T @ np.linalg.inv(S)
    P_post = (np.eye(A.shape[0]) - L @ H) @ P_pred
    return P_pred, L, P_post


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(M, dtype=float)))))


def as_matrix(x, shape=None):
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        if shape is None:
            raise ValueError("scalar needs a shape")
        return np.eye(shape[0]) * float(a) if len(shape) == 2 and shape[0] == shape[1] else np.full(shape, float(a))
    if a.ndim == 1 and shape is not None and len(shape) == 2 and shape[0] == shape[1]:
        return np.diag(a)
    return a
