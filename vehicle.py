import numpy as np


def copy(car):
    # Copy the car
    new_car = Vehicle(x=car.x, y=car.y, psi=car.psi, v=car.v, delta=car.delta, L=car.L)
    new_car.a = car.a
    new_car.omega = car.omega
    new_car.path = car.path.copy()
    return new_car


class Vehicle:
    def __init__(self, x=0, y=0, psi=0, v=0, delta=0, L=1.0):
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
        self.max_delta = np.pi / 4
        self.min_delta = -self.max_delta

        self.max_a = 1.0
        self.min_a = -self.max_a
        self.max_omega = 1.0
        self.min_omega = -self.max_omega

        self.d = 1.0  # safe distance

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

    def move_state(self, a, omega, dt):
        x = self.x + self.v * np.cos(self.psi) * dt
        y = self.y + self.v * np.sin(self.psi) * dt
        psi = self.psi + self.v * np.tan(self.delta) / self.L * dt
        v = self.v + a * dt
        delta = self.delta + omega * dt
        return x, y, psi, v, delta

    def move_delta(self, a, omega, dt):
        dx = self.v * np.cos(self.psi) * dt
        dy = self.v * np.sin(self.psi) * dt
        dpsi = self.v * np.tan(self.delta) / self.L * dt
        dv = a * dt
        ddelta = omega * dt
        return dx, dy, dpsi, dv, ddelta

    def is_goal_reached(self, pose, goal, threshold=0.1):
        # Check if the car has reached the goal
        return np.linalg.norm(pose[:2] - goal[:2]) < threshold
        pass

    # define a prediction trajectory with horizon N and input u
    def prediction_traj(self, N, u, dt):
        # The prediction trajectory
        x_traj = np.zeros((N + 1, 5))
        x_traj[0] = np.array([self.x, self.y, self.psi, self.v, self.delta])
        for i in range(N):
            x_traj[i + 1] = x_traj[i] + self.move_delta(u[i, 0], u[i, 1], dt)
        return x_traj
