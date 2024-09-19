
# Model Predictive Control using CasADi

This repository implements Model Predictive Control (MPC) for various dynamical systems using the CasADi optimization framework. The goal is to provide an efficient and flexible Python-based environment for solving control problems in robotics, autonomous vehicles, and other engineering applications.

## Features

- **CasADi Integration**: Utilizes CasADi for automatic differentiation and solving optimization problems.
- **MPC Framework**: Implements a generic MPC framework, allowing users to define system dynamics, cost functions, and constraints.
- **Trajectory Tracking**: Includes examples for solving trajectory tracking and path-following problems for car-like robots and other dynamic systems.
- **Nonlinear Dynamics**: Supports nonlinear system models for accurate simulation and control.
- **Simulations**: Provides simulation scripts for testing and validating MPC on custom models.

## Requirements

### System Requirements

- **Python**: 3.6+
- **CasADi**: 3.5.5 or later

### Python Dependencies

Install the required dependencies using the following command:
```bash
pip install casadi numpy matplotlib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NamDinhRobotics/mpc_casadi_ws.git
   cd mpc_casadi_ws
   ```

2. The repository is now ready for use.

## Usage

### MPC Example

To run an example of MPC for a simple car-like robot, navigate to the appropriate directory and run the script:

```bash
python robot_mpc_class.py
```

This will simulate a car-like robot following a predefined trajectory using MPC. The dynamics, cost functions, and constraints can be customized in the script.

### Demo
[two robots.mp4](doc/vid1.mp4)
<video src="doc/vid1.mp4" width="320" height="200" controls preload></video>

[two robots exp2.mp4](doc/multi_car_animation.mp4)
<video src="doc/multi_car_animation.mp4" width="320" height="200" controls preload></video>

### Customizing the MPC

1. **Dynamics**: Define the system dynamics in CasADi symbolic form in the `mpc_solver.py` file. Modify the state equations to suit your system.
   
2. **Cost Function**: Adjust the cost function within the MPC loop to prioritize different objectives, such as tracking error, control effort, or energy consumption.

3. **Constraints**: Add constraints such as state and control limits, collision avoidance, or other problem-specific conditions.

### Running Other Examples

The repository contains multiple examples, including trajectory optimization and dynamic obstacle avoidance. Run any of these examples using:

```bash
python examples/example_name.py
```

### Visualizing Results

Each example script includes visualization of the results using `matplotlib`. The state trajectories and control inputs are plotted at the end of each simulation for easy interpretation.

## Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and make pull requests. Please ensure your code adheres to the style of the project and includes appropriate documentation and tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
