import numpy as np

"""
a first-order system
"""

class Object():
    # Plant
    def __init__(self, delta_t):
        np.random.seed(1)
        self.x = 0.0
        self.xd = 0.0
        self.xd_smooth = np.array([0.0, 0.0])
        self.xd_filtered = 0.0
        self.u = 0.0
        self.f = 0.0

        self.e_old = 0
        self.ei = 0

        self.t = 0.0
        self.delta_t = delta_t
        self.t_last = - 10000

    def xd_update(self):
        if (self.t - self.t_last) > 5.0:
            xd_old = self.xd
            self.xd = np.random.uniform(-1, 1) * 1.0
            while np.abs(xd_old - self.xd) < 0.4:
                self.xd = np.random.uniform(-1, 1) * 1.0
            self.t_last = np.copy(self.t)

    def controller_update(self):
        kd = 0.1
        kp = 10.0
        ki = 2.0

        self.xd_smooth_update()
        e = self.xd_smooth[0] - self.x
        e = self.xd_filtered - self.x
        dot_e = (e - self.e_old) / self.delta_t

        self.u = kp * e + ki * self.ei + kd * dot_e  # PID
        self.ei += e * self.delta_t
        self.e_old = e

    def dynamic(self, x1, u):
        f = -x1 + np.arctan(u)

        return f

    def base_dynamic(self, x1, u):
        f = -0.5 * x1

        return f

    def xd_smooth_update(self):
        xi = 1
        omega = 5
        x1 = self.xd_smooth[0]
        x2 = self.xd_smooth[1]

        dot_x1 = x2
        dot_x2 = -2 * xi * omega * x2 - omega ** 2 * (x1 - self.xd)

        self.xd_smooth += np.array([dot_x1, dot_x2]) * self.delta_t

        T = 0.7
        self.xd_filtered += 1/T * (self.xd - self.xd_filtered) * self.delta_t

    def update(self):
        self.xd_update()
        self.controller_update()
        self.f = self.dynamic(self.x, self.u)
        self.f0 = self.base_dynamic(self.x, self.u)

        self.x = self.x + self.f * self.delta_t
        self.t = self.t + self.delta_t

        return self.x, self.xd, self.u, self.f
