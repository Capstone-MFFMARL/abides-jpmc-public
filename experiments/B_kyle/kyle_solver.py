import numpy as np


def solve_kyle(N, sigma_u, Sigma0, tau=1.0):
    """Solves Kyle discrete multi-period model recursively (Equations 39-44).

    Returns a dict containing:
        - lam: array of length N+1 (index 1..N used)
        - beta: array of length N+1 (index 1..N used)
        - alpha: array of length N+1 (index 0..N used)
        - Sigma: array of length N+1 (index 0..N used)
        - delta: array of length N+1 (index 0..N used)
    """
    lam = np.zeros(N + 1)
    beta = np.zeros(N + 1)
    alpha = np.zeros(N + 1)
    Sigma = np.zeros(N + 1)
    delta = np.zeros(N + 1)

    # 1. Start with normalized terminal conditions
    Sigma[N] = 1.0
    alpha[N] = 0.0
    delta[N] = 0.0

    # 2. Backward loop n = N down to 1
    for n in range(N, 0, -1):
        a_curr = alpha[n]
        s_curr = Sigma[n]
        d_curr = delta[n]

        if abs(a_curr) < 1e-14:
            coeff_a = 2.0 * tau * (sigma_u**2)
            l_val = np.sqrt(s_curr / coeff_a)
            surviving_roots = [l_val]
        else:
            coeffs = [
                -2.0 * tau * (sigma_u**2) * a_curr,
                2.0 * tau * (sigma_u**2),
                2.0 * a_curr * s_curr,
                -s_curr,
            ]
            roots = np.roots(coeffs)

            surviving_roots = []
            for r in roots:
                if np.isreal(r):
                    r_real = float(np.real(r))
                    cond1 = r_real > 0
                    cond2 = (1.0 - 2.0 * a_curr * r_real) > 0
                    cond3 = (r_real * (1.0 - a_curr * r_real)) > 0
                    if cond1 and cond2 and cond3:
                        surviving_roots.append(r_real)

        assert (
            len(surviving_roots) == 1
        ), f"Expected exactly 1 valid root at step {n}, found {len(surviving_roots)}"
        l_n = surviving_roots[0]
        lam[n] = l_n

        # Update step
        b_n = (1.0 - 2.0 * a_curr * l_n) / (2.0 * l_n * (1.0 - a_curr * l_n) * tau)
        beta[n] = b_n

        s_prev = s_curr / (1.0 - b_n * l_n * tau)
        d_prev = d_curr + a_curr * (l_n**2) * (sigma_u**2) * tau
        a_prev = 1.0 / (4.0 * l_n * (1.0 - a_curr * l_n))

        Sigma[n - 1] = s_prev
        delta[n - 1] = d_prev
        alpha[n - 1] = a_prev

    # 3. Rescale with c = Sigma0 / Sigma[0]
    c = Sigma0 / Sigma[0]
    sqrt_c = np.sqrt(c)

    lam *= sqrt_c
    beta /= sqrt_c
    alpha /= sqrt_c
    Sigma *= c
    delta *= sqrt_c

    return {
        "lam": lam,
        "beta": beta,
        "alpha": alpha,
        "Sigma": Sigma,
        "delta": delta,
    }