import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the array
x = np.array([[[0], [1], [2]]])

# Get coordinates and values
x_coords = np.arange(x.shape[1])
y_coords = np.arange(x.shape[0])
z_coords = np.arange(x.shape[2])
values = x.flatten()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c=values, cmap='viridis', s=100)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('Simple 3D Scatter Plot of Array x')
plt.show()
