import matplotlib.pyplot as plt
import numpy as np

# Load data from file
data = np.loadtxt('/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/DATA/trajectory_test.txt')

# Extract x, y, z coordinates
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a new figure for the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
scatter = ax.scatter(x, y, z, c='b', marker='o')

# Fix the scales of the axes
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_zlim(0, 50)

# Add labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Plot of Points')

# Show the plot
plt.show()
