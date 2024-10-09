import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan2, pi

# Define points and convert them into a NumPy array for easy indexing
points = np.array([(1, 13), (2, 9), (4, 4), (5, 2), (5, 5), (5, 12), (7, 7), (10, 4), (12, 9), (13, 9)])

# Function to calculate rho and theta for a line through two points
def line_parameters(x1, y1, x2, y2):
    if x2 == x1:
        theta = pi / 2  # Vertical line case
        rho = x1  # rho is the x-coordinate for vertical lines
    else:
        theta = atan2(y2 - y1, x2 - x1)
        rho = x1 * cos(theta) + y1 * sin(theta)
    return rho, theta

# Check if a point is on a line defined by rho, theta
def is_point_on_line(x, y, rho, theta, tolerance=1e-5):
    return abs(rho - (x * cos(theta) + y * sin(theta))) < tolerance

# Dictionary to hold lines and the points they pass through
line_points = {}

# Iterate over every pair of points to find lines
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        x1, y1 = points[i]
        x2, y2 = points[j]
        rho, theta = line_parameters(x1, y1, x2, y2)
        rho_rounded = round(rho, 1)
        theta_rounded = round(theta, 2)
        line_key = (rho_rounded, theta_rounded)
        if line_key not in line_points:
            line_points[line_key] = set()
        line_points[line_key].update([points[i], points[j]])

# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], color='red', label='Points')

# Plot each line found
for (rho, theta), pts in line_points.items():
    x_vals = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    if sin(theta) == 0:  # Avoid division by zero for vertical lines
        y_vals = np.repeat(rho / cos(theta), 100)
    else:
        y_vals = (rho - x_vals * cos(theta)) / sin(theta)
    plt.plot(x_vals, y_vals, label=f'Line rho={rho:.1f}, theta={theta:.2f} rad')

plt.legend()
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Lines Passing Through Points')
plt.grid(True)
plt.show()

# Output lines and their points
print("Lines and the points they pass through:")
for key, value in line_points.items():
    print(f"Line (rho={key[0]:.1f}, theta={key[1]:.2f} rad): passes through points {value}")
