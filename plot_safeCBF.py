import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_safe_region(center, P, gamma):
    # Compute the ellipse parameters
    eigvals, eigvecs = np.linalg.eig(P)
    radii = np.sqrt(eigvals) * np.sqrt(gamma)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    ellipse = Ellipse(center, width=radii[0] * 2, height=radii[1] * 2, angle=np.degrees(angle),
                      edgecolor='r', facecolor='none', linestyle='--', label='Safe Region')

    return ellipse


# Example parameters
center = np.array([5, 5])
P = np.array([[2, 0.5], [0.5, 1]])
gamma = 1

fig, ax = plt.subplots()
ellipse = plot_safe_region(center, P, gamma)
ax.add_patch(ellipse)

# Plot settings
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Safe Region Visualization')
plt.legend()
plt.grid(True)
plt.show()
