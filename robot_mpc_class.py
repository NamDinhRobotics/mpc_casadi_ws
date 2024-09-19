import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
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

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N + 1, 5))
        self.u0 = np.zeros((self.N, 2))

    def setupController(self, obs):
        self.opti = ca.Opti()
        # Control signal
        self.U = self.opti.variable(self.N, 2)
        control_a = self.U[:, 0]
        control_omega = self.U[:, 1]

        # State system
        self.X = self.opti.variable(self.N + 1, 5)
        state_x = self.X[:, 0]
        state_y = self.X[:, 1]
        state_psi = self.X[:, 2]
        state_v = self.X[:, 3]
        state_delta = self.X[:, 4]

        # create model
        f = lambda x, u: ca.vertcat(*[
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] / self.car.L * ca.tan(x[4]),
            u[0],
            u[1]
        ])

        # define target state
        self.x_ref = self.opti.parameter(2, 5)

        # define control reference N
        self.u_ref = self.opti.parameter(self.N, 2)

        # define k obstacles
        num_obs = len(obs)
        self.x_obs = self.opti.parameter(num_obs)
        self.y_obs = self.opti.parameter(num_obs)
        self.r_obs = self.opti.parameter(num_obs)

        # initial constraints x0 = self.x_ref[0]
        self.opti.subject_to(self.X[0, :] == self.x_ref[0, :])
        for i in range(self.N):
            x_next = self.X[i, :] + self.T * f(self.X[i, :], self.U[i, :]).T
            self.opti.subject_to(self.X[i + 1, :] == x_next)

        # cost function
        obj = 0
        goal_state = self.x_ref[-1, :]
        for i in range(self.N):
            obj += ca.mtimes([(self.X[i, :] - goal_state), self.Q, (self.X[i, :] - goal_state).T]) + ca.mtimes(
                [self.U[i, :], self.R, self.U[i, :].T])

        self.opti.minimize(obj)

        # define boundaries
        self.opti.subject_to(self.opti.bounded(self.car.min_v, state_v, self.car.max_v))
        self.opti.subject_to(self.opti.bounded(self.car.min_delta, state_delta, self.car.max_delta))

        self.opti.subject_to(self.opti.bounded(self.car.min_a, control_a, self.car.max_a))
        self.opti.subject_to(self.opti.bounded(self.car.min_omega, control_omega, self.car.max_omega))

        # define obstacles
        for i in range(num_obs):
            ob_x = self.x_obs[i]
            ob_y = self.y_obs[i]
            ob_r = self.r_obs[i]
            self.opti.subject_to((state_x - ob_x) ** 2 + (state_y - ob_y) ** 2 >= (ob_r + self.car.d) ** 2)

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_ref, u_ref, obs):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.u_ref, u_ref)
        self.opti.set_value(self.x_obs, obs[:, 0])
        self.opti.set_value(self.y_obs, obs[:, 1])
        self.opti.set_value(self.r_obs, obs[:, 2])

        x0 = x_ref[0, :]

        self.opti.set_initial(self.X, np.tile(x0, (self.N + 1, 1)))
        self.opti.set_initial(self.U, np.zeros((self.N, 2)))

        self.opti.solve()

        u_opt = self.opti.value(self.U)
        x_opt = self.opti.value(self.X)

        self.u0, self.next_states = shift(u_opt, x_opt)
        return u_opt[:, 0], u_opt[:, 1]

    def solve_obs(self, x_ref, u_ref, x_obs, y_obs, r_obs):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.u_ref, u_ref)
        self.opti.set_value(self.x_obs, x_obs)
        self.opti.set_value(self.y_obs, y_obs)
        self.opti.set_value(self.r_obs, r_obs)

        x0 = x_ref[0, :]
        self.opti.set_initial(self.X, np.tile(x0, (self.N + 1, 1)))
        self.opti.set_initial(self.U, np.zeros((self.N, 2)))

        self.opti.solve()

        u_opt = self.opti.value(self.U)
        x_opt = self.opti.value(self.X)

        self.u0, self.next_states = shift(u_opt, x_opt)
        return u_opt[:, 0], u_opt[:, 1]

    def is_goal_reached(self, current_state, goal_state, tolerance=0.1):
        distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        return distance < tolerance

    def run_until_goal(self, x_ref, u_ref, obs, tolerance=0.1):
        self.setupController(obs)
        current_state = x_ref[0, :]
        goal_state = x_ref[-1, :]

        trajectory = [current_state]
        control_inputs = []

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 13)
        ax.set_ylim(-1, 13)

        # Initialize plot elements
        actual_line, = ax.plot([], [], 'b-', label='Actual Trajectory', linewidth=2)
        predict_line, = ax.plot([], [], 'r--', label='Predicted Trajectory', linewidth=1)

        rect = Rectangle((0, 0), 1, 1, angle=0, color='g', fill=True, label='Robot Pose', alpha=0.2)
        ax.add_patch(rect)

        obs_circles = [plt.Circle((obs[i, 0], obs[i, 1]), obs[i, 2], color='r', fill=True, alpha=0.2) for i in range(len(obs))]
        for circle in obs_circles:
            ax.add_artist(circle)

        # Plot start and goal markers
        ax.plot(x_ref[0, 0], x_ref[0, 1], 'g*', label='Start')
        ax.plot(x_ref[-1, 0], x_ref[-1, 1], 'ro', label='Goal')

        ax.legend()

        def update_plot():
            trajectory_np = np.array(trajectory)
            actual_line.set_data(trajectory_np[:, 0], trajectory_np[:, 1])

            # Update predicted trajectory
            predicted_trajectory = self.next_states
            predict_line.set_data(predicted_trajectory[:, 0], predicted_trajectory[:, 1])

            # Update rectangle to represent the robot pose
            x, y, psi = current_state[0], current_state[1], current_state[2]
            robot_width, robot_length = 0.5, 1.0
            rear_axle_x = robot_length / 4
            rect.set_width(robot_length)
            rect.set_height(robot_width)
            rect.angle = np.degrees(psi)
            center_x = x - robot_length / 2 * np.cos(psi) + robot_width / 2 * np.sin(psi)
            center_y = y - robot_length / 2 * np.sin(psi) - robot_width / 2 * np.cos(psi)
            rx = center_x + rear_axle_x * np.cos(psi)
            ry = center_y + rear_axle_x * np.sin(psi)
            rect.set_xy((rx, ry))

            ax.relim()
            ax.autoscale_view()
            plt.draw()

        # Function to handle key press events
        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        # Connect the key press event to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)

        simulation_running = True
        while simulation_running and not self.is_goal_reached(current_state, goal_state, tolerance):
            u_a, u_omega = self.solve(x_ref, u_ref, obs)
            current_state = self.next_states[1]
            trajectory.append(current_state)
            control_inputs.append([u_a[0], u_omega[0]])  # Store only the first control input

            x_ref = np.concatenate(([current_state], x_ref[1:]))
            u_ref = np.concatenate((self.u0, u_ref[self.N:]))

            update_plot()
            plt.pause(0.1)

            if not plt.get_fignums():  # Check if the figure has been closed
                simulation_running = False

        plt.ioff()
        plt.close(fig)

        return np.array(trajectory), np.array(control_inputs)

    def plot_results(self, trajectory, control_inputs, x_ref, obs):
        time = np.arange(len(trajectory)) * self.T

        fig = plt.figure(figsize=(20, 15))
        grid = plt.GridSpec(3, 3, figure=fig)

        # Trajectory plot
        ax_traj = fig.add_subplot(grid[:, 0])
        ax_traj.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Actual Trajectory')
        ax_traj.plot(x_ref[0, 0], x_ref[0, 1], 'g*', markersize=10, label='Start')
        ax_traj.plot(x_ref[-1, 0], x_ref[-1, 1], 'r*', markersize=10, label='Goal')

        # Plot obstacles
        for ob in obs:
            circle = Circle((ob[0], ob[1]), ob[2], fill=False, color='r')
            ax_traj.add_artist(circle)

        ax_traj.set_xlabel('X Position (m)')
        ax_traj.set_ylabel('Y Position (m)')
        ax_traj.set_title('Robot Trajectory')
        ax_traj.legend()
        ax_traj.grid(True)
        ax_traj.axis('equal')

        # X and Y positions over time
        ax_x = fig.add_subplot(grid[0, 1])
        ax_x.plot(time, trajectory[:, 0], 'b-')
        ax_x.set_ylabel('X Position (m)')
        ax_x.grid(True)

        ax_y = fig.add_subplot(grid[0, 2])
        ax_y.plot(time, trajectory[:, 1], 'r-')
        ax_y.set_ylabel('Y Position (m)')
        ax_y.grid(True)

        # Orientation and velocity over time
        ax_psi = fig.add_subplot(grid[1, 1])
        ax_psi.plot(time, np.degrees(trajectory[:, 2]), 'g-')
        ax_psi.set_ylabel('Orientation (degrees)')
        ax_psi.grid(True)

        ax_v = fig.add_subplot(grid[1, 2])
        ax_v.plot(time, trajectory[:, 3], 'm-')
        ax_v.set_ylabel('Velocity (m/s)')
        ax_v.grid(True)

        # Control inputs over time
        ax_a = fig.add_subplot(grid[2, 1])
        ax_a.plot(time[:-1], control_inputs[:, 0], 'c-')
        ax_a.set_ylabel('Acceleration (m/s^2)')
        ax_a.set_xlabel('Time (s)')
        ax_a.grid(True)

        ax_omega = fig.add_subplot(grid[2, 2])
        ax_omega.plot(time[:-1], np.degrees(control_inputs[:, 1]), 'y-')
        ax_omega.set_ylabel('Steering Rate (degrees/s)')
        ax_omega.set_xlabel('Time (s)')
        ax_omega.grid(True)

        plt.tight_layout()
        plt.show()


# In the test_run function:
def test_run():
    car = Car()
    mpc_controller = MPCController(car, T=0.2, N=20)

    x_ref = np.array([[0, 0, 0, 0, 0], [10, 10, 0, 0, 0]])
    u_ref = np.zeros((mpc_controller.N, 2))
    obs = np.array([[5, 5, 1]])

    trajectory, control_inputs = mpc_controller.run_until_goal(x_ref, u_ref, obs)
    mpc_controller.plot_results(trajectory, control_inputs, x_ref, obs)


if __name__ == '__main__':
    test_run()
