import os
import matplotlib.pyplot as plt
import numpy as np

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

base_path = "/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/DATA"

with open(os.path.join(base_path, "ref_imu_data.txt")) as ref_file:
    ref_acc_data = []
    ref_gyro_data = []
    for line in ref_file:
        line = line.strip().split()
        ref_acc_data.append([float(x) for x in line[2:5]])
        ref_gyro_data.append([float(x) for x in line[5:8]])

with open(os.path.join(base_path, "real_imu_data.txt")) as real_file:
    real_acc_data = []
    real_gyro_data = []
    for line in real_file:
        line = line.strip().split()
        real_acc_data.append([float(x) for x in line[2:5]])
        real_gyro_data.append([float(x) for x in line[5:8]])

ref_acc_data = np.array(ref_acc_data)
ref_gyro_data = np.array(ref_gyro_data)
real_acc_data = np.array(real_acc_data)
real_gyro_data = np.array(real_gyro_data)

# import ipdb; ipdb.set_trace()

# Plot X coordinate
axs[0].plot(ref_acc_data[:, 0], label='Ground Truth', color='r', linestyle='solid')
axs[0].plot(real_acc_data[:, 0], label='Simulation', color='b', linestyle='dashed')
axs[0].set_title('X Coordinate')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X')
axs[0].legend()

# Plot Y coordinate
axs[1].plot(ref_acc_data[:, 1], label='Ground Truth', color='r', linestyle='solid')
axs[1].plot(real_acc_data[:, 1], label='Simulation', color='b', linestyle='dashed')
axs[1].set_title('Y Coordinate')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Y')
axs[1].legend()

# Plot Z coordinate
axs[2].plot(ref_acc_data[:, 2], label='Ground Truth', color='r', linestyle='solid')
axs[2].plot(real_acc_data[:, 2], label='Simulation', color='b', linestyle='dashed')
axs[2].set_title('Z Coordinate')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Z')
axs[2].legend()

plt.suptitle('Accelerometer Data')
plt.tight_layout()
plt.show()

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot X coordinate
axs[0].plot(ref_gyro_data[:, 0], label='Ground Truth', color='r', linestyle='solid')
axs[0].plot(real_gyro_data[:, 0], label='Simulation', color='b', linestyle='dashed')
axs[0].set_title('X Coordinate')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X')
axs[0].legend()

# Plot Y coordinate
axs[1].plot(ref_gyro_data[:, 1], label='Ground Truth', color='r', linestyle='solid')
axs[1].plot(real_gyro_data[:, 1], label='Simulation', color='b', linestyle='dashed')
axs[1].set_title('Y Coordinate')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Y')
axs[1].legend()

# Plot Z coordinate
axs[2].plot(ref_gyro_data[:, 2], label='Ground Truth', color='r', linestyle='solid')
axs[2].plot(real_gyro_data[:, 2], label='Simulation', color='b', linestyle='dashed')
axs[2].set_title('Z Coordinate')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Z')
axs[2].legend()

plt.suptitle('Gyroscope Data')
plt.tight_layout()
plt.show()