# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from tqdm import tqdm


# def plot_poses(gt, save_path):
#     """
#     Plot ground truth poses in 3D.

#     Parameters:
#         gt (list): List of ground truth poses, where each pose is a list/array of [x, y, z].
#     """
#     gt_x, gt_y, gt_z = [], [], []
#     for pose in gt:
#         gt_x.append(pose[0])
#         gt_y.append(pose[1])
#         gt_z.append(pose[2])

#     # Plotting
#     fig = plt.figure(figsize=(15, 5))

#     # 3D plot
#     ax3d = fig.add_subplot(131, projection='3d')
#     ax3d.scatter(gt_x, gt_y, gt_z, c='r', marker='^', label='Ground Truth Poses')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#     ax3d.set_zlabel('Z')
#     ax3d.set_title('3D Plot')
#     ax3d.legend()

#     # Top view
#     ax_top = fig.add_subplot(132)
#     ax_top.scatter(gt_x, gt_y, c='r', marker='^', label='Ground Truth Poses')
#     ax_top.set_xlabel('X')
#     ax_top.set_ylabel('Y')
#     ax_top.set_title('Top View')

#     # Right view
#     ax_right = fig.add_subplot(133)
#     ax_right.scatter(gt_y, gt_z, c='r', marker='^', label='Ground Truth Poses')
#     ax_right.set_xlabel('Y')
#     ax_right.set_ylabel('Z')
#     ax_right.set_title('Right View')

#     # Save plot as image
#     plt.tight_layout()
#     plt.savefig(save_path)


# if __name__ == "__main__":
#     # Read a list of folders at a path and store the names in a list
#     DATA_PATH = '/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/DATA/'
#     OUTPUT_PATH  = '/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/plots'
#     folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

#     for folder in tqdm(folders) :
#         data = np.loadtxt(DATA_PATH + folder + '/trajectory_test.txt')
#         plot_poses(data, OUTPUT_PATH + f"/{folder}_plot.jpg")


import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from matplotlib import cm

def plot_poses(gt, plot_name, save_path):
    """
    Plot ground truth poses in 3D with color indicating the elevation.

    Parameters:
        gt (list): List of ground truth poses, where each pose is a list/array of [x, y, z].
    """
    gt_x, gt_y, gt_z = [], [], []
    for pose in gt:
        gt_x.append(pose[0])
        gt_y.append(pose[1])
        gt_z.append(pose[2])

    # Create a colormap based on z values
    norm = plt.Normalize(min(gt_z), max(gt_z))
    colors = cm.viridis(norm(gt_z))  # You can choose any colormap that fits your taste

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax3d = fig.add_subplot(131, projection='3d')
    scatter3d = ax3d.scatter(gt_x, gt_y, gt_z, c=colors, marker='^', label='Ground Truth Poses')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title(plot_name)
    # fig.colorbar(scatter3d, ax=ax3d, label='Elevation (Z)')

    # Top view
    ax_top = fig.add_subplot(132)
    ax_top.scatter(gt_x, gt_y, c=colors, marker='^', label='Ground Truth Poses')
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Y')
    ax_top.set_title('Top View')
    # fig.colorbar(scatter3d, ax=ax_top, label='Elevation (Z)')

    # Right view
    ax_right = fig.add_subplot(133)
    ax_right.scatter(gt_y, gt_z, c=colors, marker='^', label='Ground Truth Poses')
    ax_right.set_xlabel('Y')
    ax_right.set_ylabel('Z')
    ax_right.set_title('Right View')
    fig.colorbar(scatter3d, ax=ax_right, label='Elevation (Z)')

    # Save plot as image
    plt.tight_layout()
    plt.savefig(save_path)

if __name__ == "__main__":
    DATA_PATH = '/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/DATA/'
    OUTPUT_PATH  = '/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/plots'
    folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
    folder_map = {
        "flat_oval": "Flat Oval",
        "bent_oval": "Rotated Oval",
        "flat_line": "Flat Line",
        "clover": "Clover",
        "fig_8": "Figure of 8",
        "bent_spiral": "Rotated Spiral",
        "wavy_circle": "Wavy Circle",
        "bent_line": "Rotated Line",
        "flat_spiral": "Flat Spiral",
    }

    for folder in tqdm(folders):
        data = np.loadtxt(DATA_PATH + folder + '/trajectory_test.txt')
        plot_3d_name = folder_map[folder]
        plot_poses(data, plot_3d_name, OUTPUT_PATH + f"/{folder}_plot.jpg")