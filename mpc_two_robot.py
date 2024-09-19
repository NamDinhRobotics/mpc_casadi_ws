import time

import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from vehicle import Vehicle


def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n


class MPCController:
    def __init__(self, car: Vehicle, T=0.2, N=10, Q=np.diag([1.0, 1.0, 0.001, 0.01, 0.01]), R=np.diag([0.01, 0.01])):
        self.car = car
        self.T = T  # time step
        self.N = N  # horizon length

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N + 1, 5))
        self.u0 = np.zeros((self.N, 2))

    def setupController(self, obs, dynamic_car_trajectories=None):
        self.opti = ca.Opti()
        # Control signal
        self.U = self.opti.variable(self.N, 2)

        self.u_opt = np.zeros((self.N, 2))
        self.x_opt = np.zeros((self.N + 1, 5))

        control_a = self.U[:, 0]
        control_omega = self.U[:, 1]

        # State system
        self.X = self.opti.variable(self.N + 1, 5)
        state_x = self.X[:, 0]
        state_y = self.X[:, 1]
        state_psi = self.X[:, 2]
        state_v = self.X[:, 3]
        state_delta = self.X[:, 4]

        # Create model
        f = lambda x, u: ca.vertcat(*[
            x[3] * ca.cos(x[2]),
            x[3] * ca.sin(x[2]),
            x[3] / self.car.L * ca.tan(x[4]),
            u[0],
            u[1]
        ])

        # Define target state
        self.x_ref = self.opti.parameter(2, 5)

        # Define control reference N
        self.u_ref = self.opti.parameter(self.N, 2)

        # Define k obstacles
        num_obs = len(obs)
        self.x_obs = self.opti.parameter(num_obs)
        self.y_obs = self.opti.parameter(num_obs)
        self.r_obs = self.opti.parameter(num_obs)

        # Define multiple dynamic cars predicted trajectories
        num_dynamic_cars = 0
        if dynamic_car_trajectories is not None:
            num_dynamic_cars = len(dynamic_car_trajectories)
            self.dynamic_cars_x = []
            self.dynamic_cars_y = []
            self.dynamic_cars_psi = []
            self.dynamic_cars_v = []
            self.dynamic_cars_delta = []
            for _ in range(num_dynamic_cars):
                self.dynamic_cars_x.append(self.opti.parameter(self.N + 1))
                self.dynamic_cars_y.append(self.opti.parameter(self.N + 1))
                self.dynamic_cars_psi.append(self.opti.parameter(self.N + 1))
                self.dynamic_cars_v.append(self.opti.parameter(self.N + 1))
                self.dynamic_cars_delta.append(self.opti.parameter(self.N + 1))

        # Initial constraints x0 = self.x_ref[0]
        self.opti.subject_to(self.X[0, :] == self.x_ref[0, :])
        for i in range(self.N):
            x_next = self.X[i, :] + self.T * f(self.X[i, :], self.U[i, :]).T
            self.opti.subject_to(self.X[i + 1, :] == x_next)

        # Cost function
        obj = 0
        goal_state = self.x_ref[-1, :]
        for i in range(self.N):
            obj += ca.mtimes([(self.X[i, :] - goal_state), self.Q, (self.X[i, :] - goal_state).T]) + ca.mtimes(
                [self.U[i, :], self.R, self.U[i, :].T])

        # Add dynamic cars avoidance cost
        for j in range(num_dynamic_cars):
            self.opti.subject_to((state_x - self.dynamic_cars_x[j]) ** 2 + (state_y - self.dynamic_cars_y[j]) ** 2 >= (0.5 + self.car.d) ** 2)
            for i in range(self.N):
                dx = state_x[i] - self.dynamic_cars_x[j][i]
                dy = state_y[i] - self.dynamic_cars_y[j][i]
                # obj += ca.exp(-0.01 * (dx ** 2 + dy ** 2))  # Add cost term to avoid dynamic car
                # Define horizon factor
                distance = ca.sqrt(dx ** 2 + dy ** 2)
                relative_velocity_x = state_v[i] * ca.cos(state_psi[i]) - self.dynamic_cars_v[j][i] * ca.cos(
                    self.dynamic_cars_psi[j][i])
                relative_velocity_y = state_v[i] * ca.sin(state_psi[i]) - self.dynamic_cars_v[j][i] * ca.sin(
                    self.dynamic_cars_psi[j][i])
                relative_velocity = ca.sqrt(relative_velocity_x ** 2 + relative_velocity_y ** 2)

                horizon_factor = (self.N - i) / self.N

                alpha_combined = (0.1 * ca.exp(-0.01 * distance)) * \
                                 (0.1 * ca.exp(-0.1 * relative_velocity)) * \
                                 (0.01 * horizon_factor)

                adaptive_cost_combined = ca.exp(-alpha_combined * distance)
                obj += adaptive_cost_combined
                self.opti.subject_to((dx ** 2 + dy ** 2) >= (0.5 + self.car.d) ** 2)  # Avoid collision

        # add constraints for car1 2
        # for i in range(num_dynamic_cars):

        self.opti.minimize(obj)

        # Define boundaries
        self.opti.subject_to(self.opti.bounded(self.car.min_v, state_v, self.car.max_v))
        self.opti.subject_to(self.opti.bounded(self.car.min_delta, state_delta, self.car.max_delta))

        self.opti.subject_to(self.opti.bounded(self.car.min_a, control_a, self.car.max_a))
        self.opti.subject_to(self.opti.bounded(self.car.min_omega, control_omega, self.car.max_omega))

        # Define obstacles
        for i in range(num_obs):
            ob_x = self.x_obs[i]
            ob_y = self.y_obs[i]
            ob_r = self.r_obs[i]
            self.opti.subject_to((state_x - ob_x) ** 2 + (state_y - ob_y) ** 2 >= (ob_r + self.car.d) ** 2)

        opts_setting = {'ipopt.max_iter': 1000,
                        'ipopt.print_level': 1,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-8}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, x_ref, u_ref, obs, dynamic_car_trajectories=None):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.u_ref, u_ref)
        self.opti.set_value(self.x_obs, obs[:, 0])
        self.opti.set_value(self.y_obs, obs[:, 1])
        self.opti.set_value(self.r_obs, obs[:, 2])

        if dynamic_car_trajectories is not None:
            for j, trajectory in enumerate(dynamic_car_trajectories):
                self.opti.set_value(self.dynamic_cars_x[j], trajectory[:, 0])
                self.opti.set_value(self.dynamic_cars_y[j], trajectory[:, 1])
                self.opti.set_value(self.dynamic_cars_psi[j], trajectory[:, 2])
                self.opti.set_value(self.dynamic_cars_v[j], trajectory[:, 3])
                self.opti.set_value(self.dynamic_cars_delta[j], trajectory[:, 4])

        x0 = x_ref[0, :]

        self.opti.set_initial(self.X, np.tile(x0, (self.N + 1, 1)))
        self.opti.set_initial(self.U, np.zeros((self.N, 2)))

        try:
            self.opti.solve()
            self.u_opt = self.opti.value(self.U)
            self.x_opt = self.opti.value(self.X)

            self.u0, self.next_states = shift(self.u_opt, self.x_opt)
            return self.x_opt, self.u_opt
        # run with exception
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            print("Solver stats:", self.opti.stats())
            print("Current state guess:", self.opti.debug.value(self.X))
            print("Current control guess:", self.opti.debug.value(self.U))
            # Handle failure case
            self.u_opt = np.zeros((self.N, 2))
            self.x_opt = np.tile(x_ref, (self.N + 1, 1))
            return self.x_opt, self.u_opt

    def planning(self, x_ref, u_ref, obs, dynamic_car_trajectory=None):
        self.setupController(obs, dynamic_car_trajectory)
        return self.solve(x_ref, u_ref, obs, dynamic_car_trajectory)

    # update the car state
    def update_car(self, a, omega):
        self.car.update_state(a, omega, self.T)
        return self.car.x, self.car.y, self.car.psi, self.car.v, self.car.delta


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def test_multi_car():
    car = Vehicle(0, 0, np.pi / 4)
    Xcar = Vehicle(0, 0, np.pi / 4)
    mpc_controller = MPCController(car, T=0.2, N=20)
    x_ref = np.array([[0, 0, np.pi / 4, 0, 0], [10, 10, 0, 0, 0]])
    u_ref = np.zeros((mpc_controller.N, 2))

    car9 = Vehicle(10, 10, np.pi / 4)
    Xcar9 = Vehicle(10, 10, np.pi / 4)
    mpc_controller9 = MPCController(car9, T=0.2, N=20)
    x_ref9 = np.array([[10, 10, np.pi / 4, 0, 0], [0, 0, np.pi / 4, 0, 0]])
    u_ref9 = np.zeros((mpc_controller9.N, 2))

    obs = np.array([[5, 5, 1.0]])

    # Plot the paths
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 13)

    line, = ax.plot([], [], 'b:', label='Trajectory', linewidth=2)
    line9, = ax.plot([], [], 'r:', label='Trajectory', linewidth=2)

    # plot line for car 1 and 2
    line1, = ax.plot([], [], 'g:', label='Trajectory', linewidth=2)
    line2, = ax.plot([], [], 'y:', label='Trajectory', linewidth=2)
    line3, = ax.plot([], [], 'y:', label='Trajectory', linewidth=2)
    line4, = ax.plot([], [], 'y:', label='Trajectory', linewidth=2)

    # Draw obstacles
    for ob in obs:
        circle = plt.Circle((ob[0], ob[1]), ob[2], color='c', fill=True, alpha=0.2)
        ax.add_artist(circle)

    # Plot start and goal markers
    ax.plot(x_ref[0, 0], x_ref[0, 1], 'g*', label='Start')  # Green star for start
    ax.plot(x_ref[-1, 0], x_ref[-1, 1], 'ro', label='Goal')  # Red circle for goal

    # Plot current car position
    rect = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='r', fill=True, label='Robot Pose', alpha=0.5)
    ax.add_patch(rect)

    rect1 = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='g', fill=True, label='Robot Pose', alpha=0.2)
    ax.add_patch(rect1)

    rect2 = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='g', fill=True, label='Robot Pose', alpha=0.2)
    ax.add_patch(rect2)

    rect3 = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='g', fill=True, label='Robot Pose', alpha=0.2)
    ax.add_patch(rect3)

    rect4 = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='g', fill=True, label='Robot Pose', alpha=0.2)
    ax.add_patch(rect4)

    rect9 = plt.Rectangle((0, 0), 0.5, 1, angle=0, color='y', fill=True, label='Robot Pose', alpha=0.5)
    ax.add_patch(rect9)

    car1 = Vehicle(2, 5, 0.0, 0.5, -0.0)
    car2 = Vehicle(5, 8, 0, -0.5, 0.0)
    car3 = Vehicle(9, 8, 0.0, -0.5, -0.0)
    car4 = Vehicle(6, 6, -np.pi / 2, 0.5, -0.0)

    metadata = dict(title='MPC Path Planning', artist='Matplotlib', comment='MPC simulation')
    writer = FFMpegWriter(fps=2, metadata=metadata)

    # save a, w for car1
    a1 = []
    w1 = []
    vel1 = []

    a99 = []
    w99 = []
    vel99 = []

    u_ref_car1 = np.zeros((mpc_controller.N, 2))
    u_ref_car2 = np.zeros((mpc_controller.N, 2))
    u_ref_car3 = np.zeros((mpc_controller.N, 2))
    u_ref_car4 = np.zeros((mpc_controller.N, 2))

    # traj
    opt_traj = np.zeros((mpc_controller.N + 1, 5))
    opt_traj9 = np.zeros((mpc_controller.N + 1, 5))
    # add current state to opt_traj with the same state
    for i in range(mpc_controller.N + 1):
        opt_traj[i] = x_ref[0]
        opt_traj9[i] = x_ref[0]

    with writer.saving(fig, "multi_car_animation.mp4", 600):
        while not mpc_controller.car.is_goal_reached(mpc_controller.car.path[-1], x_ref[-1]) or \
                not mpc_controller9.car.is_goal_reached(mpc_controller9.car.path[-1], x_ref9[-1]):
            # Example dynamic car trajectories (replace with actual predicted trajectories)

            dynamic_car_trajectories1 = []
            # check distance car1, car2. car3 , car4 to car
            if (distance([car1.x, car1.y], [car.x, car.y]) < 4):
                dynamic_car_trajectories1.append(car1.prediction_traj(mpc_controller.N, u_ref_car1, mpc_controller.T))
            if (distance([car2.x, car2.y], [car.x, car.y]) < 4):
                dynamic_car_trajectories1.append(car2.prediction_traj(mpc_controller.N, u_ref_car2, mpc_controller.T))
            if (distance([car3.x, car3.y], [car.x, car.y]) < 4):
                dynamic_car_trajectories1.append(car3.prediction_traj(mpc_controller.N, u_ref_car3, mpc_controller.T))
            if (distance([car4.x, car4.y], [car.x, car.y]) < 4):
                dynamic_car_trajectories1.append(car4.prediction_traj(mpc_controller.N, u_ref_car4, mpc_controller.T))

            if (distance([Xcar9.x, Xcar9.y], [car.x, car.y]) < 4):
                dynamic_car_trajectories1.append(Xcar9.prediction_traj(mpc_controller.N, u_ref_car4, mpc_controller.T))
            x_opt, u_opt = mpc_controller.planning(x_ref, u_ref, obs, dynamic_car_trajectories1)

            a, omega = u_opt[0]
            # add opt traj

            # add car 9
            dynamic_car_trajectories2 = []
            if (distance([car1.x, car1.y], [car9.x, car9.y]) < 4):
                dynamic_car_trajectories2.append(car1.prediction_traj(mpc_controller.N, u_ref_car1, mpc_controller.T))
            if (distance([car2.x, car2.y], [car9.x, car9.y]) < 4):
                dynamic_car_trajectories2.append(car2.prediction_traj(mpc_controller.N, u_ref_car2, mpc_controller.T))
            if (distance([car3.x, car3.y], [car9.x, car9.y]) < 4):
                dynamic_car_trajectories2.append(car3.prediction_traj(mpc_controller.N, u_ref_car3, mpc_controller.T))
            if (distance([car4.x, car4.y], [car9.x, car9.y]) < 4):
                dynamic_car_trajectories2.append(car4.prediction_traj(mpc_controller.N, u_ref_car4, mpc_controller.T))

            if (distance([Xcar.x, Xcar.y], [car9.x, car9.y]) < 4):
                dynamic_car_trajectories2.append(Xcar.prediction_traj(mpc_controller.N, u_ref_car4, mpc_controller.T))

            x_opt9, u_opt9 = mpc_controller9.planning(x_ref9, u_ref9, obs, dynamic_car_trajectories2)
            a9, omega9 = u_opt9[0]

            # save a and omega
            a1.append(a)
            w1.append(omega)
            vel1.append(mpc_controller.car.v)

            a99.append(a9)
            w99.append(omega9)
            vel99.append(mpc_controller9.car.v)

            x, y, psi, v, delta = mpc_controller.update_car(a, omega)
            x9, y9, psi9, v9, delta9 = mpc_controller9.update_car(a9, omega9)

            # Update x_ref for the next iteration
            x_ref[0, :] = [x, y, psi, v, delta]
            u_ref = np.concatenate((mpc_controller.u0, u_ref[mpc_controller.N:]))
            x_ref9[0, :] = [x9, y9, psi9, v9, delta9]
            u_ref9 = np.concatenate((mpc_controller9.u0, u_ref9[mpc_controller9.N:]))

            traj1 = car1.prediction_traj(mpc_controller.N, u_ref_car1, mpc_controller.T)
            traj2 = car2.prediction_traj(mpc_controller.N, u_ref_car2, mpc_controller.T)
            traj3 = car3.prediction_traj(mpc_controller.N, u_ref_car3, mpc_controller.T)
            traj4 = car4.prediction_traj(mpc_controller.N, u_ref_car4, mpc_controller.T)

            line1.set_data(traj1[:, 0], traj1[:, 1])
            line2.set_data(traj2[:, 0], traj2[:, 1])
            line3.set_data(traj3[:, 0], traj3[:, 1])
            line4.set_data(traj4[:, 0], traj4[:, 1])

            # Update dynamic cars' state
            car1.update_state(0.0, 0.0, 1 * mpc_controller.T)
            car2.update_state(-0.0, 0.0, 1 * mpc_controller.T)
            car3.update_state(-0.0, -0.0, 1 * mpc_controller.T)
            car4.update_state(0.0, -0.0, 1 * mpc_controller.T)

            Xcar.update_state(a, omega, mpc_controller.T)
            Xcar9.update_state(a9, omega9, mpc_controller.T)

            # Set line data from x_opt, take the current state
            line.set_data(x_opt[:, 0], x_opt[:, 1])
            line9.set_data(x_opt9[:, 0], x_opt9[:, 1])

            # plot line for car 1 and 2, predict trajectory

            # Plot the car
            robot_width, robot_length = 0.5, 1.0  # Adjust size as necessary
            rear_axle_x = robot_length / 4
            rect.set_width(robot_length)
            rect.set_height(robot_width)
            rect.angle = np.degrees(psi)
            # Compute the center of the rectangle with angle psi from traj1
            x = x_opt[0, 0]
            y = x_opt[0, 1]

            center_x = x - robot_length / 2 * np.cos(psi) + robot_width / 2 * np.sin(psi)
            center_y = y - robot_length / 2 * np.sin(psi) - robot_width / 2 * np.cos(psi)
            # Adjust to rear center
            rx = center_x + rear_axle_x * np.cos(psi)
            ry = center_y + rear_axle_x * np.sin(psi)
            rect.set_xy((rx, ry))

            rect9.set_width(robot_length)
            rect9.set_height(robot_width)
            rect9.angle = np.degrees(psi9)
            # Compute the center of the rectangle with angle psi from traj1
            x9 = x_opt9[0, 0]
            y9 = x_opt9[0, 1]

            center_x9 = x9 - robot_length / 2 * np.cos(psi9) + robot_width / 2 * np.sin(psi9)
            center_y9 = y9 - robot_length / 2 * np.sin(psi9) - robot_width / 2 * np.cos(psi9)
            # Adjust to rear center
            rx9 = center_x9 + rear_axle_x * np.cos(psi9)
            ry9 = center_y9 + rear_axle_x * np.sin(psi9)
            rect9.set_xy((rx9, ry9))

            # plot car1, car2
            x1, y1, psi1, v1, delta1, del1, om1 = car1.path[-1]
            x2, y2, psi2, v2, delta2, del2, om2 = car2.path[-1]
            x3, y3, psi3, v3, delta3, del3, om3 = car3.path[-1]
            x4, y4, psi4, v4, delta4, del4, om4 = car4.path[-1]

            rect1.set_width(robot_length)
            rect1.set_height(robot_width)
            rect1.angle = np.degrees(psi1)
            center_x1 = x1 - robot_length / 2 * np.cos(psi1) + robot_width / 2 * np.sin(psi1)
            center_y1 = y1 - robot_length / 2 * np.sin(psi1) - robot_width / 2 * np.cos(psi1)
            rx1 = center_x1 + rear_axle_x * np.cos(psi1)
            ry1 = center_y1 + rear_axle_x * np.sin(psi1)
            rect1.set_xy((rx1, ry1))

            rect2.set_width(robot_length)
            rect2.set_height(robot_width)
            rect2.angle = np.degrees(psi2)
            center_x2 = x2 - robot_length / 2 * np.cos(psi2) + robot_width / 2 * np.sin(psi2)
            center_y2 = y2 - robot_length / 2 * np.sin(psi2) - robot_width / 2 * np.cos(psi2)
            rx2 = center_x2 + rear_axle_x * np.cos(psi2)
            ry2 = center_y2 + rear_axle_x * np.sin(psi2)
            rect2.set_xy((rx2, ry2))

            rect3.set_width(robot_length)
            rect3.set_height(robot_width)
            rect3.angle = np.degrees(psi3)
            center_x3 = x3 - robot_length / 2 * np.cos(psi3) + robot_width / 2 * np.sin(psi3)
            center_y3 = y3 - robot_length / 2 * np.sin(psi3) - robot_width / 2 * np.cos(psi3)
            rx3 = center_x3 + rear_axle_x * np.cos(psi3)
            ry3 = center_y3 + rear_axle_x * np.sin(psi3)
            rect3.set_xy((rx3, ry3))

            rect4.set_width(robot_length)
            rect4.set_height(robot_width)
            rect4.angle = np.degrees(psi4)
            center_x4 = x4 - robot_length / 2 * np.cos(psi4) + robot_width / 2 * np.sin(psi4)
            center_y4 = y4 - robot_length / 2 * np.sin(psi4) - robot_width / 2 * np.cos(psi4)
            rx4 = center_x4 + rear_axle_x * np.cos(psi4)
            ry4 = center_y4 + rear_axle_x * np.sin(psi4)
            rect4.set_xy((rx4, ry4))

            ax.relim()
            ax.autoscale_view()
            plt.draw()

            writer.grab_frame()
            # save plt as gif
            # Update the figure
            plt.pause(0.01)
            try:
                writer.grab_frame()
            except Exception as e:
                print(f"Error grabbing frame: {e}")
            time.sleep(0.01)  # Simulate real-time update
    plt.ioff()  # Turn off interactive mode
    plt.show()
    # create new fig and plot a, w
    fig, ax = plt.subplots()
    ax.plot(a1, label='a')
    ax.plot(w1, label='w')
    ax.plot(vel1, label='v')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    test_multi_car()
