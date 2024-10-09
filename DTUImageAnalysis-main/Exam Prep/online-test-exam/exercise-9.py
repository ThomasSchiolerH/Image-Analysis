import numpy as np
import math

# Coordinates of the dots
coordinates = np.array([[0, 1], [0, 5], [2, 4], [4, 4], [3, 6]])

# Function to calculate rho and theta
def calculate_hough_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    theta = math.atan2((y2 - y1), (x2 - x1))
    rho = x1 * math.cos(theta) + y1 * math.sin(theta)
    theta = math.degrees(theta)  # Convert to degrees
    return [theta, rho]

# Define the lines of the matchstick person
lines = [((3,2), (3,6)),  # Body
         ((2,4), (3,5)),  # Left arm
         ((4,4), (3,5)),  # Right arm
         ((3,2), (5,0)),  # Right leg
         ((3,2), (1,0))]  # Left leg

# Calculate Hough transform for each line
hough_lines = [calculate_hough_line(i, j) for i, j in lines]

# Print lines
for line in hough_lines:
    print(f"Theta: {line[0]:.2f}, Rho: {line[1]:.2f}")


