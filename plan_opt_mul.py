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
        Q = ca.diagcat(15, 15, 0.001)  # State cost matrix
        R = ca.diagcat(1, 0.5)  # Control cost matrix
        obj = 0  # Objective function
        g = []  # Constraints

        g.append(X[:, 0] - P[:self.n_states])

        # Formulate the NLP
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            # obj += ca.mtimes([(st - P[self.n_states:]).T, Q, (st - P[self.n_states:])]) + ca.mtimes([con.T, R, con])
            # only use position to reference position
            obj += 0.1 * (st[0] - P[self.n_states]) ** 2 + 0.1 * (st[1] - P[self.n_states + 1]) ** 2


            # Add penalty to avoid zero velocity
            obj += 0.1 * (con[0] - 0.1) ** 2

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

    def set_obstacle(self, x_obs, y_obs, r_obs):
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.r_obs = r_obs

    def set_safe_margin(self, safety_margin):
        self.safety_margin = safety_margin

    def solve(self, x0, x_goal, other_robot_poses, num_attempts=10, perturbation_scale=0.1):
        self.set_optimization_problem(self.x_obs, self.y_obs, self.r_obs, other_robot_poses, self.safety_margin)

        best_solution = None
        best_cost = np.inf

        for attempt in range(num_attempts):
            # Initial guess for the decision variables with added perturbation
            X0 = np.zeros((self.n_states, self.N + 1))
            U0 = np.zeros((self.n_controls, self.N))
            X0 += perturbation_scale * np.random.randn(*X0.shape)
            U0 += perturbation_scale * np.random.randn(*U0.shape)

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

            try:
                # Solve the problem
                sol = self.solver(x0=ca.vertcat(X0.reshape((-1, 1)), U0.reshape((-1, 1))),
                                  lbx=lbx, ubx=ubx,
                                  lbg=lbg, ubg=ubg,
                                  p=p)

                # Extract the solution
                x_sol = sol['x'].full().flatten()
                x_state = x_sol[:self.n_states * (self.N + 1)].reshape((self.n_states, self.N + 1))
                u_control = x_sol[self.n_states * (self.N + 1):].reshape((self.n_controls, self.N))

                cost = sol['f'].full().flatten()[0]

                # Update the best solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (x_state, u_control)

            except RuntimeError as e:
                print(f"Attempt {attempt + 1}/{num_attempts} failed: {e}")

        return best_solution


# Example usage
model = KinematicBicycleModel(L=2.0, T=1, N=50, v_min=-1.0, v_max=1.0, delta_min=-0.5, delta_max=0.5)
model.set_obstacle([5, 10], [5, 10], [1, 1])
model.set_safe_margin(0.1)
x0 = [0, 0, 0]  # Initial state
x_goal = [10, 10, 0]  # Goal state
other_robot_poses = [[5, 5], [7, 7]]  # Other robots
solution = model.solve(x0, x_goal, other_robot_poses)
if solution is not None:
    x_state, u_control = solution
    plt.plot(x_state[0, :], x_state[1, :], 'b')
    plt.plot([x0[0], x_goal[0]], [x0[1], x_goal[1]], 'ro')
    plt.show()
else:
    print("No solution found")
