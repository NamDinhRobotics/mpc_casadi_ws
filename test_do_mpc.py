import numpy as np
from casadi import vertcat, sin, cos
import do_mpc
import matplotlib.pyplot as plt
import os


# Define model type
model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Define state variables
x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
y = model.set_variable(var_type='_x', var_name='y', shape=(1, 1))
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
v = model.set_variable(var_type='_x', var_name='v', shape=(1, 1))

# Define control inputs
accel = model.set_variable(var_type='_u', var_name='accel')
steer = model.set_variable(var_type='_u', var_name='steer')

# Define the dynamics
L = 2.0  # Length of the car
dx = v * cos(theta)
dy = v * sin(theta)
dtheta = v * steer / L
dv = accel

model.set_rhs('x', dx)
model.set_rhs('y', dy)
model.set_rhs('theta', dtheta)
model.set_rhs('v', dv)

model.setup()

# Define the MPC Controller
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 2,
    'store_full_solution': True,
 #   'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}
mpc.set_param(**setup_mpc)

# Objective Function
goal = np.array([10, 10])
mterm = (model.x['x'] - goal[0])**2 + (model.x['y'] - goal[1])**2
lterm = (model.x['x'] - goal[0])**2 + (model.x['y'] - goal[1])**2

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(accel=1e-2, steer=1e-2)

# State and Control Constraints
mpc.bounds['lower', '_x', 'v'] = 0
mpc.bounds['upper', '_x', 'v'] = 10
mpc.bounds['lower', '_u', 'accel'] = -1
mpc.bounds['upper', '_u', 'accel'] = 1
mpc.bounds['lower', '_u', 'steer'] = -0.5
mpc.bounds['upper', '_u', 'steer'] = 0.5

# Obstacle Avoidance
obstacles = [(5, 5, 1), (7, 8, 1)]  # Each tuple is (x, y, radius)

for (ox, oy, r) in obstacles:
    obstacle_constraint = (model.x['x'] - ox)**2 + (model.x['y'] - oy)**2
    mpc.set_nl_cons(f'obs_{ox}_{oy}', obstacle_constraint, ub=(r**2))

# Setup the MPC controller
mpc.setup()

# Simulation
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.1)
simulator.setup()

x0 = np.array([[0], [0], [0], [0]])
simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

# Run simulation until the goal is reached
path = []
for _ in range(200):  # Adjust the number of steps if necessary
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
    path.append(x0)
    if np.linalg.norm(x0[:2] - goal[:, None]) < 0.5:  # Goal tolerance
        break

# Visualization
path = np.array(path).squeeze()

plt.plot(path[:, 0], path[:, 1], label='Path')
for (ox, oy, r) in obstacles:
    circle = plt.Circle((ox, oy), r, color='r', fill=False)
    plt.gca().add_artist(circle)
plt.scatter([goal[0]], [goal[1]], color='g', label='Goal')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
