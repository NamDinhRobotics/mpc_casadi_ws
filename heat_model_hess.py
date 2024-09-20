import numpy as np
import matplotlib.pyplot as plt


class RealisticHeaterModel:
    def __init__(self, C=5000, eta=0.9, h=0.1, T_amb=25, delta_t=1.0):
        self.C = C        # Thermal capacity (more realistic)
        self.eta = eta    # Heater efficiency (more realistic)
        self.h = h        # Heat loss coefficient (realistic)
        self.T_amb = T_amb  # Ambient temperature (constant for simplicity)
        self.delta_t = delta_t  # Time step (seconds)
        self.T_out = T_amb  # Initial temperature
        self.noise_level_input = 0.2  # Reduced input disturbance
        self.noise_level_output = 0.5  # Increased noise in output measurement
        self.temperature = T_amb  # Actual temperature (unknown to EKF)

    def step(self, u):
        """Update temperature for one time step given PWM input u (0-100%)."""
        # Input noise (e.g., disturbance from environment)
        d_in = np.random.randn() * self.noise_level_input
        # Heat input from PWM (u is percentage)
        P = self.eta * u
        # Update the temperature based on the dynamics
        self.T_out += self.delta_t / self.C * (P - self.h * (self.T_out - self.T_amb) + d_in)

        # Measurement noise
        d_out = np.random.randn() * self.noise_level_output
        # Measured temperature
        T_meas = self.T_out + d_out
        self.temperature = T_meas
        return T_meas


class EKFUnknownHeatModel:
    def __init__(self, alpha=0.1, t_est =25, delta_t=1.0):
        # EKF parameters
        self.alpha = alpha  # PWM to temperature gain (unknown, so assumed)
        self.T_est = t_est  # Initial temperature estimate
        self.P = 1.0  # Initial covariance estimate
        self.Q = 0.2  # Process noise covariance
        self.R = 0.5  # Measurement noise covariance

    def state_transition(self, T, u):
        """Simple assumed model: temperature increases/decreases based on PWM input."""
        return T + self.alpha * u

    def measurement_model(self, T):
        """Measurement model: temperature is measured directly with noise."""
        return T

    def ekf_predict(self, u):
        """Predict the next temperature estimate based on control input."""
        # Predict the next state (temperature)
        T_pred = self.state_transition(self.T_est, u)

        # Jacobian of the state transition (since T_new = T + alpha * u, F = 1)
        F = 1

        # Predict the covariance
        self.P = F * self.P * F + self.Q

        return T_pred

    def ekf_update(self, T_meas, T_pred):
        """Update the estimate using the measurement."""
        # Measurement residual (innovation)
        y_residual = T_meas - self.measurement_model(T_pred)

        # Jacobian of the measurement function (identity matrix, H = 1)
        H = 1

        # Kalman gain
        K = self.P * H / (H * self.P * H + self.R)

        # Update the state estimate
        self.T_est = T_pred + K * y_residual

        # Update the covariance
        self.P = (1 - K * H) * self.P

    def step(self, u, T_meas):
        """Perform one EKF step (prediction + update)."""
        # Predict the next state
        T_pred = self.ekf_predict(u)

        # Update the estimate based on the measurement
        self.ekf_update(T_meas, T_pred)

        return self.T_est


class FeedbackOptimizationController:
    def __init__(self, target_temp, learning_rate=0.1, hessian=1.0, u_min=0, u_max=100):
        self.target_temp = target_temp  # Desired target temperature
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.hessian = hessian  # Second-order information (Hessian)
        self.u_min = u_min  # Minimum control input (0%)
        self.u_max = u_max  # Maximum control input (100%)
        self.u = 50.0  # Initial control input (PWM, start at 50%)

        # Tracking history for plotting
        self.history_u = []
        self.history_temp = []
        self.history_error = []

    def gradient(self, temp):
        """Gradient of the objective function: error between measured temp and target."""
        return temp - self.target_temp  # Gradient is the error (T - T_target)

    def step(self, temp_measured):
        """Perform one feedback optimization step using Newton's method."""
        # Compute the gradient of the cost function
        grad = self.gradient(temp_measured)

        # Newton's update step with gradient and Hessian
        self.u -= self.learning_rate * grad / self.hessian

        # compute the Hessian using the second derivative of the cost function

        # Clamp the control input to be within valid bounds [u_min, u_max]
        self.u = np.clip(self.u, self.u_min, self.u_max)

        # Track history for plotting
        self.history_u.append(self.u)
        self.history_temp.append(temp_measured)
        self.history_error.append(temp_measured - self.target_temp)

        return self.u

    def plot_results(self):
        """Plot the control input, temperature, and error over time."""
        plt.figure(figsize=(12, 5))

        # Plot control input (PWM signal)
        plt.subplot(1, 3, 1)
        plt.plot(self.history_u, label='Control Input (PWM)')
        plt.title("Control Input (PWM) Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("PWM (%)")
        plt.grid(True)

        # Plot temperature over time
        plt.subplot(1, 3, 2)
        plt.plot(self.history_temp, label='Measured Temperature')
        plt.axhline(self.target_temp, color='r', linestyle='--', label='Target Temperature')
        plt.title("Temperature Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)

        # Plot error over time
        plt.subplot(1, 3, 3)
        plt.plot(self.history_error, label='Error')
        plt.title("Error Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Temperature Error (°C)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Simulation of the heat system with EKF and Feedback Optimization Controller
n_steps = 10000  # Number of time steps
dt = 2.0  # Time step

# Create instances of the heat system, EKF, and feedback optimization controller
heat_system = RealisticHeaterModel(C=5000, eta=0.9, h=0.1, T_amb=35, delta_t=dt)
ekf_heater = EKFUnknownHeatModel(alpha=0.015, t_est=35)  # EKF uses a slightly different assumed alpha
controller = FeedbackOptimizationController(target_temp=60, learning_rate=2.0)

T_actual = []  # Actual temperature (unknown to EKF)
T_measured = []  # Measured temperature (with noise)
T_estimated = []  # EKF estimated temperature

# Run the simulation
for t in range(n_steps):
    # Get the actual measured temperature from the heat system
    T_meas = heat_system.step(controller.u)

    # Use EKF to estimate the temperature
    T_est = ekf_heater.step(controller.u, T_meas)

    # Use the controller to adjust the PWM based on the estimated temperature
    controller.step(T_est)

    # Store results for plotting
    T_actual.append(heat_system.temperature)
    T_measured.append(T_meas)
    T_estimated.append(T_est)

# Plot the results of feedback optimization controller and EKF
controller.plot_results()

# Plot the actual, measured, and EKF-estimated temperature
plt.figure(figsize=(10, 6))
plt.plot(range(n_steps), T_actual, label='Actual Temperature (unknown model)', color='blue')
plt.plot(range(n_steps), T_measured, label='Measured Temperature (with noise)', color='red', linestyle='--')
plt.plot(range(n_steps), T_estimated, label='EKF Estimated Temperature', color='green', linestyle='-.')
plt.xlabel('Time (steps)')
plt.ylabel('Temperature (°C)')
plt.title('EKF Temperature Estimation and Feedback Optimization for Heat System')
plt.legend()
plt.grid(True)
plt.show()
