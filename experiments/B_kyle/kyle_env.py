import numpy as np


class KyleGame:

    def __init__(
        self,
        M,
        N,
        sigma_u,
        v_mean=1000.0,
        v_std=100.0,
        p0=1000.0,
        batch=400000,
        tau=1.0,
    ):
        self.M = M
        self.N = N
        self.sigma_u = sigma_u
        self.v_mean = v_mean
        self.v_std = v_std
        self.p0 = p0
        self.batch = batch
        self.tau = tau
        self.reset()

    def reset(self, v_fixed=None, p0_fixed=None):
        if v_fixed is not None:
            self.v = np.full(self.batch, v_fixed, dtype=np.float64)
        else:
            self.v = np.random.normal(
                self.v_mean, self.v_std, size=self.batch
            ).astype(np.float64)

        current_p0 = p0_fixed if p0_fixed is not None else self.p0
        self.p = np.full(self.batch, current_p0, dtype=np.float64)
        self.step_idx = 0
        return self.p.copy()

    def step(self, x, depth_params):
        if np.isscalar(depth_params):
            lam_arr = np.full(self.M, depth_params, dtype=np.float64)
        else:
            lam_arr = np.asarray(depth_params, dtype=np.float64)

        u = np.random.normal(
            0.0, self.sigma_u * np.sqrt(self.tau), size=self.batch
        )
        q = x + u

        # Split q across M market makers pro-rata by depth d_i = 1 / |lambda_i|
        depths = 1.0 / np.abs(lam_arr)
        total_depth = np.sum(depths)

        # VWAP calculation
        lam_eff = np.sum(np.sign(lam_arr)) / total_depth
        p_vwap = self.p + lam_eff * q

        # Price clipping
        p_new = np.clip(p_vwap, 0.5 * self.v, 1.5 * self.v)

        # Rewards
        r_insider = x * (self.v - p_new)
        r_mm = q * (p_new - self.p)

        self.p = p_new
        self.step_idx += 1

        return p_new, r_insider, r_mm, q, lam_eff