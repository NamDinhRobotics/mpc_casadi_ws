import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n

class Car:
    def __init__(self):
        # Vehicle parameters
        self.L = 2.5  # Wheelbase length in meters
        self.d = 0.2  # Safety margin in meters

        # Control input constraints
        self.min_a = -2.0  # Minimum acceleration (m/s^2)
        self.max_a = 2.0   # Maximum acceleration (m/s^2)
        self.min_omega = -np.radians(30)  # Minimum steering rate (rad/s)
        self.max_omega = np.radians(30)   # Maximum steering rate (rad/s)

        # State constraints
        self.min_v = 0.0    # Minimum velocity (m/s)
        self.max_v = 5.0   # Maximum velocity (m/s)
        self.min_delta = -np.radians(30)  # Minimum steering angle (rad)
        self.max_delta = np.radians(30)   # Maximum steering angle (rad)

class MPCControllerFixedGamma:
    def __init__(self, car: Car, T=0.1, N=50, gamma=0.2, Q=np.diag([100.0, 100.0, 1.0, 1.0, 1.0]), R=np.diag([0.1, 0.1])):
        self.car = car
        self.T = T  # time step
        self.N = N  # horizon length
        self.gamma = gamma  # Fixed gamma value
        self.Q = Q
        self.R = R
        self.next_states = np.zeros((self.N + 1, 5))
        self.u0 = np.zeros((self.N, 2))

    def control_barrier_function(self, x, y, obs_x, obs_y, obs_r):
        safety_margin = 0.1  # Additional safety margin
        return (x - obs_x) ** 2 + (y - obs_y) ** 2 - (obs_r + self.car.d + safety_margin) ** 2

    def setupController(self, obs, use_cbf=True):
        self.opti = ca.Opti()
        self.U = self.opti.variable(self.N, 2)
        control_a = self.U[:, 0]
        control_omega = self.U[:, 1]

        self.X = self.opti.variable(self.N + 1, 5)
        state_x = self.X[:, 0]
        state_y = self.X[:, 1]
        state_psi = self.X[:, 2]
        state_v = self.X[:, 3]
        state_delta = self.X[:, 4]

        f = lambda x, u: ca.vertcat(*[
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] / self.car.L * ca.tan(x[4]),
            u[0],
            u[1]
        ])

        self.x_ref = self.opti.parameter(2, 5)
        self.u_ref = self.opti.parameter(self.N, 2)

        num_obs = len(obs)
        self.x_obs = self.opti.parameter(num_obs)
        self.y_obs = self.opti.parameter(num_obs)
        self.r_obs = self.opti.parameter(num_obs)

        self.opti.subject_to(self.X[0, :] == self.x_ref[0, :])
        for i in range(self.N):
            x_next = self.X[i, :] + self.T * f(self.X[i, :], self.U[i, :]).T
            self.opti.subject_to(self.X[i + 1, :] == x_next)

        obj = 0
        goal_state = self.x_ref[-1, :]
        for i in range(self.N):
            obj += ca.mtimes([(self.X[i, :] - goal_state), self.Q, (self.X[i, :] - goal_state).T]) + \
                   ca.mtimes([self.U[i, :], self.R, self.U[i, :].T])

        self.opti.minimize(obj)

        # Bounds on states and controls
        self.opti.subject_to(self.opti.bounded(self.car.min_v, state_v, self.car.max_v))
        self.opti.subject_to(self.opti.bounded(self.car.min_delta, state_delta, self.car.max_delta))
        self.opti.subject_to(self.opti.bounded(self.car.min_a, control_a, self.car.max_a))
        self.opti.subject_to(self.opti.bounded(self.car.min_omega, control_omega, self.car.max_omega))

        # CBF constraints with fixed gamma
        for i in range(self.N):
            for j in range(num_obs):
                h = self.control_barrier_function(state_x[i], state_y[i], self.x_obs[j], self.y_obs[j], self.r_obs[j])
                h_next = self.control_barrier_function(state_x[i + 1], state_y[i + 1], self.x_obs[j], self.y_obs[j], self.r_obs[j])
                if use_cbf:
                    self.opti.subject_to(h_next - h + self.gamma * h >= 0)
                else:
                    self.opti.subject_to(h_next >= 0)

        opts_setting = {'ipopt.max_iter': 5000,
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

        # Initial guesses
        self.opti.set_initial(self.X, np.tile(x0, (self.N + 1, 1)))
        self.opti.set_initial(self.U, np.zeros((self.N, 2)))

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
        except RuntimeError:
            # If solver fails, return previous control and state
            print("Solver failed, using previous control inputs.")
            u_opt = self.u0
            x_opt = self.next_states

        self.u0, self.next_states = shift(u_opt, x_opt)
        return u_opt[:, 0], u_opt[:, 1]

    # def is_goal_reached(self, current_state, goal_state, tolerance=0.5, speed_threshold=0.1):
    #     distance = np.linalg.norm(current_state[:2] - goal_state[:2])
    #     speed = current_state[3]
    #     return distance < tolerance and speed < speed_threshold
    def is_goal_reached(self, current_state, goal_state, tolerance=0.1):
        distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        return distance < tolerance

    def run_until_goal(self, x_ref, u_ref, obs, tolerance=0.5, use_cbf=True):
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
            plt.pause(0.001)

        # Function to handle key press events
        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        # Connect the key press event to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)

        simulation_running = True
        while simulation_running and not self.is_goal_reached(current_state, goal_state, tolerance):
            self.setupController(obs, use_cbf=use_cbf)
            u_a, u_omega = self.solve(x_ref, u_ref, obs)
            current_state = self.next_states[1]
            trajectory.append(current_state)
            control_inputs.append([u_a[0], u_omega[0]])  # Store only the first control input

            x_ref = np.concatenate(([current_state], x_ref[1:]))
            u_ref = np.concatenate((self.u0, u_ref[self.N:]))

            update_plot()

            if not plt.get_fignums():  # Check if the figure has been closed
                simulation_running = False

        plt.ioff()
        plt.close(fig)

        return np.array(trajectory), np.array(control_inputs)
# The code for the Car class and shift function remains the same as in Method 1

class MPCControllerAdaptiveGamma:
    def __init__(self, car: Car, T=0.1, N=50, gamma_desired=0.2, Q=np.diag([1000.0, 1000.0, 1.0, 1.0, 1.0]), R=np.diag([0.1, 0.1])):
        self.gamma = None
        self.car = car
        self.T = T  # time step
        self.N = N  # horizon length
        self.gamma_desired = gamma_desired  # Desired gamma value
        self.Q = Q
        self.R = R
        self.next_states = np.zeros((self.N + 1, 5))
        self.u0 = np.zeros((self.N, 2))

    def control_barrier_function(self, x, y, obs_x, obs_y, obs_r):
        safety_margin = 0.1  # Additional safety margin
        return (x - obs_x) ** 2 + (y - obs_y) ** 2 - (obs_r + self.car.d + safety_margin) ** 2

    def setupController(self, obs, use_cbf=True, gamma_desired=0.2):
        self.opti = ca.Opti()
        self.U = self.opti.variable(self.N, 2)
        control_a = self.U[:, 0]
        control_omega = self.U[:, 1]

        self.X = self.opti.variable(self.N + 1, 5)
        state_x = self.X[:, 0]
        state_y = self.X[:, 1]
        state_psi = self.X[:, 2]
        state_v = self.X[:, 3]
        state_delta = self.X[:, 4]

        # Declare gamma as an optimization variable
        self.gamma = self.opti.variable(self.N)

        f = lambda x, u: ca.vertcat(*[
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] / self.car.L * ca.tan(x[4]),
            u[0],
            u[1]
        ])

        self.x_ref = self.opti.parameter(2, 5)
        self.u_ref = self.opti.parameter(self.N, 2)

        num_obs = len(obs)
        self.x_obs = self.opti.parameter(num_obs)
        self.y_obs = self.opti.parameter(num_obs)
        self.r_obs = self.opti.parameter(num_obs)

        self.opti.subject_to(self.X[0, :] == self.x_ref[0, :])
        for i in range(self.N):
            x_next = self.X[i, :] + self.T * f(self.X[i, :], self.U[i, :]).T
            self.opti.subject_to(self.X[i + 1, :] == x_next)

        obj = 0
        goal_state = self.x_ref[-1, :]
        for i in range(self.N):
            # Objective function includes penalty on gamma deviation
            obj += ca.mtimes([(self.X[i, :] - goal_state), self.Q, (self.X[i, :] - goal_state).T]) + \
                   ca.mtimes([self.U[i, :], self.R, self.U[i, :].T]) + \
                   1 * (self.gamma[i] - gamma_desired) ** 2  # Weight for gamma deviation

        self.opti.minimize(obj)

        # Bounds on states and controls
        self.opti.subject_to(self.opti.bounded(self.car.min_v, state_v, self.car.max_v))
        self.opti.subject_to(self.opti.bounded(self.car.min_delta, state_delta, self.car.max_delta))
        self.opti.subject_to(self.opti.bounded(self.car.min_a, control_a, self.car.max_a))
        self.opti.subject_to(self.opti.bounded(self.car.min_omega, control_omega, self.car.max_omega))

        # Bounds on gamma
        gamma_min = 0.01
        gamma_max = 0.2
        self.opti.subject_to(self.opti.bounded(gamma_min, self.gamma, gamma_max))

        # add delta_gamma constraints
        for i in range(self.N - 1):
            self.opti.subject_to(self.gamma[i + 1] - self.gamma[i] <= 0.1)
            self.opti.subject_to(self.gamma[i + 1] - self.gamma[i] >= -0.1)

        # CBF constraints with adaptive gamma
        for i in range(self.N):
            for j in range(num_obs):
                h = self.control_barrier_function(state_x[i], state_y[i], self.x_obs[j], self.y_obs[j], self.r_obs[j])
                h_next = self.control_barrier_function(state_x[i + 1], state_y[i + 1], self.x_obs[j], self.y_obs[j], self.r_obs[j])
                if use_cbf:
                    self.opti.subject_to(h_next - h + self.gamma[i] * h >= 0)
                    # h(x) is small, gamma should be small, and vice versa

                else:
                    self.opti.subject_to(h_next >= 0)


        opts_setting = {'ipopt.max_iter': 3000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-6,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_ref, u_ref, obs):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.u_ref, u_ref)
        self.opti.set_value(self.x_obs, obs[:, 0])
        self.opti.set_value(self.y_obs, obs[:, 1])
        self.opti.set_value(self.r_obs, obs[:, 2])

        x0 = x_ref[0, :]

        # Initial guesses
        self.opti.set_initial(self.X, np.tile(x0, (self.N + 1, 1)))
        self.opti.set_initial(self.U, np.zeros((self.N, 2)))
        self.opti.set_initial(self.gamma, np.full(self.N, self.gamma_desired))

        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U)
            x_opt = sol.value(self.X)
            gamma_opt = sol.value(self.gamma)
        except RuntimeError:
            # If solver fails, return previous control and state
            print("Solver failed, using previous control inputs.")
            u_opt = self.u0
            x_opt = self.next_states
            gamma_opt = np.full(self.N, self.gamma_desired)

        self.u0, self.next_states = shift(u_opt, x_opt)
        return u_opt[:, 0], u_opt[:, 1], gamma_opt

    # def is_goal_reached(self, current_state, goal_state, tolerance=0.5, speed_threshold=0.1):
    #     distance = np.linalg.norm(current_state[:2] - goal_state[:2])
    #     speed = current_state[3]
    #     return distance < tolerance and speed < speed_threshold
    def is_goal_reached(self, current_state, goal_state, tolerance=0.1):
        distance = np.linalg.norm(current_state[:2] - goal_state[:2])
        return distance < tolerance

    def run_until_goal(self, x_ref, u_ref, obs, tolerance=0.5, use_cbf=True):
        current_state = x_ref[0, :]
        goal_state = x_ref[-1, :]

        trajectory = [current_state]
        control_inputs = []
        gamma_values = []

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
            plt.pause(0.1)

        # Function to handle key press events
        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        # Connect the key press event to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)

        simulation_running = True
        safety_margins = []
        while simulation_running and not self.is_goal_reached(current_state, goal_state, tolerance):
            gamma_ref = self.gamma_desired
            # decrease gamma if the robot is close to the obstacle
            # dis_obs = np.linalg.norm(current_state[:2] - obs[0, :2])
            # # linearly decrease gamma from 0.5 to 0.01 as the robot gets closer to the obstacle
            # if dis_obs < 5.1:
            #     # 5 is the distance threshold to start decreasing gamma
            #     # dis = 5 --> gamma = 0.2, dis = 1 --> gamma = 0.01
            #     # more small distance to obs, more small gamma
            #     gamma_ref = 0.2 - (0.2 - 0.01) * (dis_obs - 1) / 4 # dis = 5, gamma = 0.2, dis = 1, gamma = 0.01
            #

            self.setupController(obs, use_cbf=use_cbf, gamma_desired = gamma_ref)
            u_a, u_omega, gamma_opt = self.solve(x_ref, u_ref, obs)
            current_state = self.next_states[1]
            trajectory.append(current_state)
            control_inputs.append([u_a[0], u_omega[0]])  # Store only the first control input
            gamma_values.append(gamma_opt[0])  # Store the first gamma value

            x_ref = np.concatenate(([current_state], x_ref[1:]))
            u_ref = np.concatenate((self.u0, u_ref[self.N:]))

            h_values = [self.control_barrier_function(current_state[0], current_state[1], ob[0], ob[1], ob[2]) for ob in
                        obs]
            min_h = min(h_values)
            safety_margins.append(min_h)

            update_plot()

            if not plt.get_fignums():  # Check if the figure has been closed
                simulation_running = False

        plt.ioff()
        plt.close(fig)

        # After simulation
        plt.figure()
        time = np.arange(len(safety_margins)) * self.T
        plt.plot(time, safety_margins, 'b-', label='Safety Margin (h)')
        plt.xlabel('Time (s)')
        plt.ylabel('Safety Margin')
        plt.title('Safety Margin Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        return np.array(trajectory), np.array(control_inputs), np.array(gamma_values)

def test_run():
    car = Car()

    # Simulation parameters
    T = 0.1
    N = 20
    gamma_fixed = 0.1  # Fixed gamma value for Method 1
    gamma_desired = 0.1  # Desired gamma value for Method 2

    # Reference trajectory and initial control inputs
    x_ref = np.array([[0, 0, 0, 0, 0], [10, 10, 0, 0, 0]])
    u_ref = np.zeros((N, 2))
    obs = np.array([[5, 5, 2]])

    # Method 1: MPC with fixed gamma
    mpc_controller_no = MPCControllerFixedGamma(car, T=T, N=N, gamma=gamma_fixed)
    trajectory_no, control_inputs_no = mpc_controller_no.run_until_goal(
        x_ref, u_ref, obs, tolerance=0.5, use_cbf=False)

    # Method 1: MPC with fixed gamma
    mpc_controller_fixed = MPCControllerFixedGamma(car, T=T, N=N, gamma=gamma_fixed)
    trajectory_fixed, control_inputs_fixed = mpc_controller_fixed.run_until_goal(
        x_ref, u_ref, obs, tolerance=0.5, use_cbf=True)

    # Method 2: MPC with adaptive gamma
    mpc_controller_adaptive = MPCControllerAdaptiveGamma(car, T=T, N=N, gamma_desired=gamma_desired)
    trajectory_adaptive, control_inputs_adaptive, gamma_values_adaptive = mpc_controller_adaptive.run_until_goal(
        x_ref, u_ref, obs, tolerance=0.5, use_cbf=True)

    # Plotting the trajectories
    plt.figure(figsize=(12, 8))
    plt.plot(trajectory_no[:, 0], trajectory_no[:, 1], 'k:', label='No CBF')
    plt.plot(trajectory_fixed[:, 0], trajectory_fixed[:, 1], 'b-', label='Fixed Gamma')
    plt.plot(trajectory_adaptive[:, 0], trajectory_adaptive[:, 1], 'r--', label='Adaptive Gamma')

    plt.plot(x_ref[0, 0], x_ref[0, 1], 'g*', markersize=10, label='Start')
    plt.plot(x_ref[-1, 0], x_ref[-1, 1], 'ro', markersize=10, label='Goal')

    # Plot obstacle
    for ob in obs:
        circle = plt.Circle((ob[0], ob[1]), ob[2], color='gray', fill=True, alpha=0.3)
        plt.gca().add_artist(circle)

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Comparison of Trajectories: Fixed Gamma vs Adaptive Gamma')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Plot gamma values for adaptive method
    time = np.arange(len(gamma_values_adaptive)) * T
    plt.figure()
    plt.plot(time, gamma_values_adaptive, 'k-', label='Adaptive Gamma')
    plt.axhline(gamma_desired, color='r', linestyle='--', label='Desired Gamma')
    plt.xlabel('Time (s)')
    plt.ylabel('Gamma')
    plt.title('Gamma Values over Time (Adaptive Gamma)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    test_run()