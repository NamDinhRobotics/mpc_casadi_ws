import numpy as np


class Car:
    def __init__(self, x=0, y=0, psi=0, v=0, delta=0, L=2.5):
        # The configuration of Car
        self.x = x  # x position
        self.y = y  # y position
        self.psi = psi  # yaw angle
        self.v = v  # velocity
        self.delta = delta  # steering angle

        self.a = 0  # acceleration
        self.omega = 0  # steering angle rate

        self.L = L  # Wheelbase

        # The paths
        self.path = [np.array([self.x, self.y, self.psi, self.v, self.delta, self.a, self.omega])]

        # The constraints of the car
        self.max_v = 1.5
        self.min_v = -self.max_v
        self.max_delta = np.pi / 6
        self.min_delta = -self.max_delta

        self.max_a = 1.5
        self.min_a = -self.max_a
        self.max_omega = 0.5
        self.min_omega = -self.max_omega

        self.d = 0.3

    def update_state(self, a, omega, dt):
        # The dynamic equations
        self.x += self.v * np.cos(self.psi) * dt
        self.y += self.v * np.sin(self.psi) * dt
        self.psi += self.v * np.tan(self.delta) / self.L * dt
        self.v += a * dt
        self.delta += omega * dt

        self.a = a
        self.omega = omega

        # Add current configuration to paths
        self.path.append(np.array([self.x, self.y, self.psi, self.v, self.delta, self.a, self.omega]))
        return self.x, self.y, self.psi, self.v, self.delta
