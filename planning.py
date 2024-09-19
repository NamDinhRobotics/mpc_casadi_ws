import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class KinematicBicycleModel:
    def __init__(self, L, T, N, v_min, v_max, delta_min, delta_max):
        self.solver = None
        self.nlp_prob = None
        self.safety_margin = None
        self.other_robot_poses = None
        self.r_obs = None
        self.y_obs = None
        self.x_obs = None
        self.f = None
        self.n_controls = None
        self.controls = None
        self.n_states = None
        self.states = None
        self.L = L  # Wheelbase
        self.T = T  # Time horizon
        self.N = N  # Number of control intervals
        self.v_min = v_min
        self.v_max = v_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.define_model()

    def define_model(self):
        # Define state and control variables
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        self.states = ca.vertcat(x, y, theta)
        self.n_states = self.states.size1()

        v = ca.MX.sym('v')
        delta = ca.MX.sym('delta')
        self.controls = ca.vertcat(v, delta)
        self.n_controls = self.controls.size1()

        # Define the kinematic model equations
        rhs = ca.vertcat(v * ca.cos(theta),
                         v * ca.sin(theta),
                         (v / self.L) * ca.tan(delta))

        # Define a function that represents the car's dynamics
        self.f = ca.Function('f', [self.states, self.controls], [rhs])

    def set_optimization_problem(self, x_obs, y_obs, r_obs, other_robot_poses, safety_margin):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs
        self.other_robot_poses = other_robot_poses
        self.safety_margin = safety_margin

        # Define the decision variables for the optimization
        U = ca.MX.sym('U', self.n_controls, self.N)
        X = ca.MX.sym('X', self.n_states, self.N + 1)
        P = ca.MX.sym('P', self.n_states + self.n_states)  # initial state + reference state

        # Define the objective function
        Q = ca.diagcat(100, 100, 0.1)  # State cost matrix
        R = ca.diagcat(0.1, 0.01)  # Control cost matrix [vel, steer]
        obj = 0  # Objective function
        g = []  # Constraints

        ## Define state error
        #    theta_des = np.arctan2(model.tvp['y_set_point'] - model.x['x', 1], model.tvp['x_set_point'] - model.x['x', 0])
        #    X = SX.zeros(3, 1)
        #    X[0] = model.x['x', 0] - model.tvp['x_set_point']
        #    X[1] = model.x['x', 1] - model.tvp['y_set_point']
        #    X[2] = np.arctan2(sin(theta_des - model.x['x', 2]), cos(theta_des - model.x['x', 2]))

        g.append(X[:, 0] - P[:self.n_states])

        # Formulate the NLP
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            #Xt = ca.MX.zeros(3, 1)
            #Pt = P[self.n_states:]
            #Xt[0] = st[0] - Pt[0]
            #Xt[1] = st[1] - Pt[1]
            #theta_des = np.arctan2(Pt[1] - Pt[0], Pt[0] - Pt[1])
            #Xt[2] = np.arctan2(ca.sin(theta_des - st[2]), ca.cos(theta_des - st[2]))
            # Compute the cost function
            #obj += ca.mtimes([Xt.T, Q, Xt]) + ca.mtimes([con.T, R, con])

            obj += ca.mtimes([(st - P[self.n_states:]).T, Q, (st - P[self.n_states:])]) + ca.mtimes([con.T, R, con])

            # Add penalty to avoid zero velocity
            # obj += 1 * (con[0] + 0.1) ** 2
            # Penalize large steering angles to avoid sharp turns
            obj += 10 * con[1] ** 2
            # Penalize paths that are close to obstacles
            for m in range(len(x_obs)):
                distance_to_obstacle = ca.sqrt((st[0] - x_obs[m]) ** 2 + (st[1] - y_obs[m]) ** 2)
                # obj += 1 * ca.exp(-distance_to_obstacle + r_obs[m] + safety_margin)

            # Smoothness penalty on control inputs
            # smoothness_penalty = ca.diagcat(0.1, 0.1)
            # obj += ca.mtimes([(U[:, 1:] - U[:, :-1]).T, smoothness_penalty, (U[:, 1:] - U[:, :-1])])

            # Distance to goal penalty
            # obj += 0.1 * ca.sumsqr(st[:2] - P[self.n_states:self.n_states + 2])

            st_next = X[:, k + 1]
            # compute next using RK4
            k1 = self.f(st, con)
            k2 = self.f(st + (self.T / (2 * self.N)) * k1, con)
            k3 = self.f(st + (self.T / (2 * self.N)) * k2, con)
            k4 = self.f(st + (self.T / self.N) * k3, con)
            st_next_rk4 = st + (self.T / self.N) * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            g.append(st_next - st_next_rk4)
        for k in range(self.N):
            st = X[:, k]
            # multiple obstacles avoidance constraints using for loop in range obstacles
            for m in range(len(x_obs)):
                g.append(ca.sqrt((st[0] - x_obs[m]) ** 2 + (st[1] - y_obs[m]) ** 2) - r_obs[m] - safety_margin)
        for k in range(self.N):
            st = X[:, k]
            # Avoidance constraints for other robots with safety margin
            for other_pose in self.other_robot_poses:
                g.append(ca.sqrt((st[0] - other_pose[0]) ** 2 + (st[1] - other_pose[1]) ** 2) - self.safety_margin)

        # Define the optimization variables and problem
        opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        self.nlp_prob = {'f': obj, 'x': opt_variables, 'g': ca.vertcat(*g), 'p': P}

        # Solver options
        opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 1000, 'ipopt.tol': 1e-8, 'ipopt.acceptable_tol': 1e-8}
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, opts)

    # set obstacle
    def set_obstacle(self, x_obs, y_obs, r_obs):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs

    def set_safe_margin(self, safety_margin):
        self.safety_margin = safety_margin

    def solve(self, x0, x_goal, other_robot_poses):
        self.set_optimization_problem(self.x_obs, self.y_obs, self.r_obs, other_robot_poses, self.safety_margin)

        X0 = np.zeros((self.n_states, self.N + 1))  # Initial guess for the states
        U0 = np.zeros((self.n_controls, self.N))  # Initial guess for the controls

        # Set initial guess for X and U
        X0[:, 0] = x0
        # X0[:, -1] = x_goal
        # U0[:, :] = 0.0
        # Set the parameters for the solver
        p = np.concatenate((x0, x_goal))

        # Adjust bounds for the optimization variables
        lbx = np.full((self.n_states * (self.N + 1) + self.n_controls * self.N), -np.inf)
        ubx = np.full((self.n_states * (self.N + 1) + self.n_controls * self.N), np.inf)

        # Set bounds for control inputs
        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.N * self.n_controls:2] = self.v_min
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.N * self.n_controls:2] = self.v_max

        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.N * self.n_controls:2] = self.delta_min
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.N * self.n_controls:2] = self.delta_max

        # Constraints for dynamics, obstacle avoidance with len, and robot collision avoidance
        lbg = np.zeros(
            (self.n_states * (self.N + 1) + (self.N * len(self.x_obs) + self.N * len(self.other_robot_poses))))
        ubg = np.zeros(
            (self.n_states * (self.N + 1) + (self.N * len(self.x_obs) + self.N * len(self.other_robot_poses))))

        # Adjust bounds for obstacle and other robot avoidance constraints
        lbg[self.n_states * (self.N + 1):] = 0  # Ensure the constraint is >= 0
        ubg[self.n_states * (self.N + 1):] = np.inf  # No upper limit on the distance

        # Solve the problem
        sol = self.solver(x0=ca.vertcat(X0.reshape((-1, 1)), U0.reshape((-1, 1))),
                          lbx=lbx, ubx=ubx,
                          lbg=lbg, ubg=ubg,
                          p=p)

        # Extract the solution
        x_sol = sol['x'].full().flatten()
        x_opt = x_sol[:self.n_states * (self.N + 1)].reshape((self.N + 1, self.n_states))
        u_opt = x_sol[self.n_states * (self.N + 1):].reshape((self.N, self.n_controls))

        return x_opt, u_opt


def update_state(state, v, delta, dt, L):
    x, y, theta = state

    # RK4 integration
    k1x, k1y, k1theta = vehicle_model(x, y, theta, v, delta, L)
    k2x, k2y, k2theta = vehicle_model(x + dt / 2 * k1x, y + dt / 2 * k1y, theta + dt / 2 * k1theta, v, delta, L)
    k3x, k3y, k3theta = vehicle_model(x + dt / 2 * k2x, y + dt / 2 * k2y, theta + dt / 2 * k2theta, v, delta, L)
    k4x, k4y, k4theta = vehicle_model(x + dt * k3x, y + dt * k3y, theta + dt * k3theta, v, delta, L)

    x_next = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_next = y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    theta_next = theta + dt / 6 * (k1theta + 2 * k2theta + 2 * k3theta + k4theta)

    return np.array([x_next, y_next, theta_next])


def vehicle_model(x, y, theta, v, delta, L):
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dtheta = v / L * np.tan(delta)

    return dx, dy, dtheta


class Robot:
    def __init__(self, L, T, N, v_min, v_max, delta_min, delta_max, x_obs, y_obs, r_obs, safety_margin):
        self.model = KinematicBicycleModel(L, T, N, v_min, v_max, delta_min, delta_max)
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs
        # set obstacle
        self.model.set_obstacle(x_obs, y_obs, r_obs)
        # set safe margin
        self.model.set_safe_margin(safety_margin)

        self.safety_margin = safety_margin
        self.pose = None
        self.goal = None

    def set_robot_obs(self, x_obs, y_obs, r_obs):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs
        self.model.set_obstacle(x_obs, y_obs, r_obs)

    def set_pose(self, pose):
        self.pose = pose

    def set_goal(self, goal):
        self.goal = goal

    def get_pose(self):
        return self.pose

    # compute next state

    def plan_path(self, other_robots):
        other_poses = [robot.get_pose() for robot in other_robots if robot != self]
        x_opt, u_opt = self.model.solve(self.pose, self.goal, other_poses)
        return x_opt, u_opt

    def run_to_goal(self, other_robots, x_obs, y_obs, r_obs, tolerance=0.2):
        trajectory = []
        input_control = []
        fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size
        while np.linalg.norm(self.pose[:2] - self.goal[:2]) > tolerance:
            # Set closest obstacle within 3m and in front of the robot
            x_obs0 = []
            y_obs0 = []
            r_obs0 = []
            obs_ind = []
            for k in range(len(x_obs)):
                #if np.linalg.norm(self.pose[:2] - np.array([x_obs[k], y_obs[k]])) < 5 and np.dot(
                #        np.array([x_obs[k], y_obs[k]]) - self.pose[:2],
                #        np.array([np.cos(self.pose[2]), np.sin(self.pose[2])])) > 0:
                if np.linalg.norm(self.pose[:2] - np.array([x_obs[k], y_obs[k]])) < 5:
                    x_obs0.append(x_obs[k] + np.random.uniform(0.0, 0.0))
                    y_obs0.append(y_obs[k] + np.random.uniform(0.0, 0.0))
                    r_obs0.append(r_obs[k] + np.random.uniform(0.0, 0.0))
                    obs_ind.append(k)
                    # break
            self.set_robot_obs(x_obs0, y_obs0, r_obs0)

            x_opt, u_opt = self.plan_path(other_robots)
            input_control.append(u_opt[0, :])

            # Update the state of the robot using RK4
            self.pose = update_state(self.pose, u_opt[0, 0], u_opt[0, 1], self.model.T / self.model.N, self.model.L)

            trajectory.append(self.pose)

            # Clear previous plot and set new plot limits
            ax.clear()
            ax.set_xlim(-0, 11)
            ax.set_ylim(-0, 11)
            # Plot the current state and goal, and the obstacles
            plt.clf()
            # set plot limits
            plt.xlim(-0, 12)
            plt.ylim(-0, 12)

            plt.plot(x_opt[:, 0], x_opt[:, 1], label='Planned path', color='r', linestyle='--')

            # Plot robot as a rectangle
            robot_length = 1.2  # Example length of the robot
            robot_width = 0.3  # Example width of the robot
            robot_outline = np.array([
                [-robot_length / 2, -robot_width / 2],
                [robot_length / 2, -robot_width / 2],
                [robot_length / 2, robot_width / 2],
                [-robot_length / 2, robot_width / 2]
            ])

            # Transform the robot outline based on current pose
            rot = np.array([
                [np.cos(self.pose[2]), -np.sin(self.pose[2])],
                [np.sin(self.pose[2]), np.cos(self.pose[2])]
            ])
            transformed_outline = np.dot(robot_outline, rot.T)
            transformed_outline[:, 0] += self.pose[0]
            transformed_outline[:, 1] += self.pose[1]

            plt.plot(transformed_outline[:, 0], transformed_outline[:, 1], 'b', label='Robot')
            # plot a rectangle for the robot transformed_outline
            plt.fill(transformed_outline[:, 0], transformed_outline[:, 1], 'b', alpha=0.3)

            #plot blue point at center of robot
            plt.plot(self.pose[0], self.pose[1], 'bo', label='Center of Robot', alpha=0.3)

            # Plot the robot's field of view as a triangle
            fov_length = 5.0  # Length of the field of view triangle (assumed for visualization)
            fov_half_angle = np.pi / 6  # Half-angle of the field of view

            fov_outline = np.array([
                [0, 0],
                [fov_length * np.cos(fov_half_angle), fov_length * np.sin(fov_half_angle)],
                [fov_length * np.cos(-fov_half_angle), fov_length * np.sin(-fov_half_angle)],
                [0, 0]
            ])

            # Transform the field of view based on current pose
            transformed_fov = np.dot(fov_outline, rot.T)
            transformed_fov[:, 0] += self.pose[0]
            transformed_fov[:, 1] += self.pose[1]

            plt.plot(transformed_fov[:, 0], transformed_fov[:, 1], 'm', label='Field of View', alpha=0.3, linestyle=':')

            # draw a yellow circle around the robot
            circle = plt.Circle((self.pose[0], self.pose[1]), 1, color='y', alpha=0.1)
            plt.gca().add_patch(circle)

            plt.plot(self.goal[0], self.goal[1], 'ro', label='Goal')
            for k in range(len(self.x_obs)):
                circle = plt.Circle((self.x_obs[k], self.y_obs[k]), self.r_obs[k], color='r', alpha=0.3)
                if obs_ind[k] == 0:
                    circle = plt.Circle((self.x_obs[k], self.y_obs[k]), self.r_obs[k], color='r', alpha=0.3)
                elif obs_ind[k] == 1:
                    circle = plt.Circle((self.x_obs[k], self.y_obs[k]), self.r_obs[k], color='g', alpha=0.3)
                elif obs_ind[k] == 2:
                    circle = plt.Circle((self.x_obs[k], self.y_obs[k]), self.r_obs[k], color='b', alpha=0.3)
                # add text obs_ind in the center of the circle
                plt.text(self.x_obs[k], self.y_obs[k], str(obs_ind[k] + 1), fontsize=30, color='red')
                plt.gca().add_patch(circle)

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.pause(0.01)  # Pause to visualize the update

        plt.show()
        return np.array(trajectory), np.array(input_control)


# Testing the Robot class
def test_robot():
    L = 2.5  # Wheelbase
    T = 1  # Time horizon
    N = 20  # Number of control intervals
    x_obs = [3.5, 6.5, 8.0]  # x-coordinates of the obstacles
    y_obs = [3.5, 6.5, 8.0]  # y-coordinate of the obstacle center
    r_obs = [1.0, 1.5, 1.0]  # Radius of the obstacle
    safety_margin = 0.6  # Safety margin to avoid other robots

    v_min = -2.5  # Minimum velocity
    v_max = 2.5  # Maximum velocity
    delta_min = -np.pi / 6  # Minimum steering angle
    delta_max = np.pi / 6  # Maximum steering angle

    robot = Robot(L, T, N, v_min, v_max, delta_min, delta_max, x_obs, y_obs, r_obs, safety_margin)

    initial_pose = np.array([11, 11, 3.14])
    goal_pose = np.array([0, 0, 3.14])
    other_robots = []

    robot.set_pose(initial_pose)
    robot.set_goal(goal_pose)

    trajectory, control = robot.run_to_goal(other_robots, x_obs, y_obs, r_obs, tolerance=0.1)

    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    plt.scatter(x_obs, y_obs, color='r', s=100, label='Obstacle')

    #plot start and gold point
    plt.plot(initial_pose[0], initial_pose[1], 'go', label='Start')
    plt.plot(goal_pose[0], goal_pose[1], 'ro', label='Goal')
    # plot obs
    for k in range(len(x_obs)):
        circle = plt.Circle((x_obs[k], y_obs[k]), r_obs[k], color='r', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    # another plot for control input
    plt.figure()
    plt.plot(control[:, 0], label='Velocity')
    plt.plot(control[:, 1], label='Steering angle')
    plt.xlabel('Time step')
    plt.ylabel('Control input')
    plt.legend(['Velocity', 'Steering angle'])
    plt.show()


# Run the test
test_robot()
