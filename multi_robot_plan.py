import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from Car import Car


def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n


class MPCController:
    def __init__(self, car: Car, T=0.1, N=50, Q=np.diag([100.0, 100.0, 1.0, 1.0, 1.0]), R=np.diag([0.1, 0.1])):
        self.car = car
        self.T = T  # time step
        self.N = N  # horizon length

        # Weight matrices
        self.Q = Q
        self.R = R

        # Initialize history states and controls
        self.next_states = np.zeros((self.N + 1, 5))
        self.u0 = np.zeros((self.N, 2))

    def setupController(self, obs, num_robots):
        self.opti = ca.Opti()

        # Control signal variables
        self.U = self.opti.variable(self.N, 2)
        control_a = self.U[:, 0]
        control_omega = self.U[:, 1]

        # State variables
        self.X = self.opti.variable(num_robots, self.N + 1, 5)
        state_x = self.X[:, :, 0]
        state_y = self.X[:, :, 1]
        state_psi = self.X[:, :, 2]
        state_v = self.X[:, :, 3]
        state_delta = self.X[:, :, 4]

        # Define model dynamics
        f = lambda x, u: ca.vertcat(*[
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] / self.car.L * ca.tan(x[4]),
            u[0],
            u[1]
        ])

        # Define target state and control references
        self.x_refs = [self.opti.parameter(2, 5) for _ in range(num_robots)]
        self.u_refs = [self.opti.parameter(self.N, 2) for _ in range(num_robots)]

        # Define obstacles
        num_obs = len(obs)
        self.x_obs = self.opti.parameter(num_obs)
        self.y_obs = self.opti.parameter(num_obs)
        self.r_obs = self.opti.parameter(num_obs)

        # Initial constraints
        for i in range(num_robots):
            self.opti.subject_to(self.X[i, 0, :] == self.x_refs[i][0, :])
            for t in range(self.N):
                x_next = self.X[i, t, :] + self.T * f(self.X[i, t, :], self.U[t, :]).T
                self.opti.subject_to(self.X[i, t + 1, :] == x_next)

        # Define cost function
        obj = 0
        for i in range(num_robots):
            goal_state = self.x_refs[i][-1, :]
            for t in range(self.N):
                obj += ca.mtimes([(self.X[i, t, :] - goal_state), self.Q, (self.X[i, t, :] - goal_state).T]) + \
                       ca.mtimes([self.U[t, :], self.R, self.U[t, :].T])
        self.opti.minimize(obj)

        # Define constraints
        for i in range(num_robots):
            self.opti.subject_to(self.opti.bounded(self.car.min_v, state_v[i, :], self.car.max_v))
            self.opti.subject_to(self.opti.bounded(self.car.min_delta, state_delta[i, :], self.car.max_delta))
            self.opti.subject_to(self.opti.bounded(self.car.min_a, control_a, self.car.max_a))
            self.opti.subject_to(self.opti.bounded(self.car.min_omega, control_omega, self.car.max_omega))

        # Define obstacle constraints
        for i in range(num_obs):
            ob_x = self.x_obs[i]
            ob_y = self.y_obs[i]
            ob_r = self.r_obs[i]
            for j in range(num_robots):
                self.opti.subject_to((state_x[j, :] - ob_x) ** 2 + (state_y[j, :] - ob_y) ** 2 >= (ob_r + self.car.d) ** 2)

        # Define collision avoidance constraints
        collision_radius = 1.0
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                for t in range(self.N):
                    self.opti.subject_to(
                        (state_x[i, t] - state_x[j, t]) ** 2 + (state_y[i, t] - state_y[j, t]) ** 2 >= collision_radius ** 2
                    )

        # Solver settings
        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}
        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_refs, u_refs, obs):
        results = []
        for i in range(len(x_refs)):
            self.opti.set_value(self.x_refs[i], x_refs[i])
            self.opti.set_value(self.u_refs[i], u_refs[i])
            self.opti.set_value(self.x_obs, obs[:, 0])
            self.opti.set_value(self.y_obs, obs[:, 1])
            self.opti.set_value(self.r_obs, obs[:, 2])

            x0 = x_refs[i][0, :]
            self.opti.set_initial(self.X[i, :, :], np.tile(x0, (self.N + 1, 1)))
            self.opti.set_initial(self.U, np.zeros((self.N, 2)))

            self.opti.solve()

            u_opt = self.opti.value(self.U)
            x_opt = self.opti.value(self.X[i, :, :])

            self.u0, next_states = shift(u_opt, x_opt)
            results.append((u_opt[:, 0], u_opt[:, 1], next_states))

        return results

    def is_goal_reached(self, current_state, goal_state, tolerance=0.1):
        distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        return distance < tolerance

    def run_until_goal(self, x_refs, u_refs, obs, tolerance=0.1):
        num_robots = len(x_refs)
        self.setupController(obs, num_robots)

        # Initialize states and goals
        states = [x_refs[i][0, :] for i in range(num_robots)]
        goals = [x_refs[i][-1, :] for i in range(num_robots)]

        trajectories = [[] for _ in range(num_robots)]

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 13)
        ax.set_ylim(-1, 13)

        # Initialize plot elements
        lines = [ax.plot([], [], line_style, label=f'Trajectory {i+1}', linewidth=2)[0] for i, line_style in enumerate(['b-', 'g--'])]
        rects = [Rectangle((0, 0), 1, 1, angle=0, color=color, fill=True, label=f'Robot Pose {i+1}', alpha=0.5) for i, color in enumerate(['b', 'g'])]
        for rect in rects:
            ax.add_patch(rect)

        obs_circles = [plt.Circle((obs[i, 0], obs[i, 1]), obs[i, 2], color='r', fill=True, alpha=0.2) for i in range(len(obs))]
        for circle in obs_circles:
            ax.add_artist(circle)

        def update_plot():
            for i in range(num_robots):
                # Update trajectories
                x_vals, y_vals = zip(*[trajectory[:2] for trajectory in trajectories[i]])
                lines[i].set_data(x_vals, y_vals)

                # Update robot pose
                x, y, psi = states[i][:3]
                robot_length = self.car.length
                robot_width = self.car.width
                rear_axle_x = 0.5 * robot_length
                center_x = x + robot_length / 2 * np.cos(psi) + robot_width / 2 * np.sin(psi)
                center_y = y - robot_length / 2 * np.sin(psi) - robot_width / 2 * np.cos(psi)
                rx = center_x + rear_axle_x * np.cos(psi)
                ry = center_y + rear_axle_x * np.sin(psi)
                rects[i].set_xy((rx, ry))
                rects[i].set_angle(np.degrees(psi))

            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)  # Pause to allow the plot to update

        while not all(self.is_goal_reached(state, goal, tolerance) for state, goal in zip(states, goals)):
            results = self.solve(x_refs, u_refs, obs)
            for i in range(num_robots):
                u_a, u_omega, next_states = results[i]
                states[i] = next_states[1]
                trajectories[i].append(states[i])

                x_refs[i] = np.concatenate(([states[i]], x_refs[i][1:]))
                u_refs[i] = np.concatenate((self.u0, u_refs[i][self.N:]))

            update_plot()

        plt.ioff()  # Turn off interactive mode
        plt.show()

        return trajectories


def test_run():
    car = Car()
    mpc_controller1 = MPCController(car)
    mpc_controller2 = MPCController(car)

    x_ref1 = np.array([[0, 0, 0, 0, 0], [10, 10, 0, 0, 0]])
    x_ref2 = np.array([[10, 10, 0, 0, 0], [0, 0, 0, 0, 0]])

    u_ref = np.zeros((mpc_controller1.N, 2))

    obs = np.array([[3, 4, 2], [5, 3, 2], [6, 7, 2]])

    trajectory1 = mpc_controller1.run_until_goal([x_ref1], [u_ref], obs)
    trajectory2 = mpc_controller2.run_until_goal([x_ref2], [u_ref], obs)


if __name__ == '__main__':
    test_run()
