import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


##################### PyTorch3D Functions ##########################

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


#######################################################################################

########################################### ATE #######################################

def calculate_ate(ground_truth_trajectory, predicted_trajectory):
    errors = []

    # Iterate over poses
    for gt_pose, pred_pose in zip(ground_truth_trajectory, predicted_trajectory):
        # Calculate position error
        gt_pos = np.array([gt_pose[0], gt_pose[1], gt_pose[2]])
        gt_ori = np.array([gt_pose[3], gt_pose[4], gt_pose[5]])
        pred_pos = np.array([pred_pose[0], pred_pose[1], pred_pose[2]])
        pred_ori = np.array([pred_pose[3], pred_pose[4], pred_pose[5]])

        pos_error = euclidean_distance(gt_pos, pred_pos)
        
        # Calculate orientation error
        ori_error = angular_distance(gt_ori, pred_ori)
        
        # Combine position and orientation errors
        pose_error = combine_errors(pos_error, ori_error)

        # Store the pose error
        errors.append(pose_error)

    # Calculate the overall ATE
    ate = calculate_overall_ate(np.array(errors))

    return ate

def euclidean_distance(position_gt, position_pred):
    # Calculate Euclidean distance between positions
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(position_gt, position_pred)))

def angular_distance(orientation_gt, orientation_pred):
    # Calculate angular distance between Euler angles
    # Assuming Euler angles are represented as (roll, pitch, yaw)
    abs_diff = [abs(a - b) for a, b in zip(orientation_gt, orientation_pred)]
    
    # Wrap angles to the range [-pi, pi]
    abs_diff_wrapped = [abs(diff - 2 * np.pi) if diff > np.pi else abs(diff) for diff in abs_diff]
    
    # Calculate the overall angular distance (root mean square error)
    angular_dist = np.sqrt(sum(diff ** 2 for diff in abs_diff_wrapped) / len(abs_diff_wrapped))
    
    return angular_dist

def combine_errors(pos_error, ori_error):
    # Combine position and orientation errors into a single scalar value
    return np.sqrt(pos_error ** 2 + ori_error ** 2)

def calculate_overall_ate(errors):
    # Calculate the overall Absolute Trajectory Error
    return np.mean(errors), np.median(errors)  # or any other aggregation method (e.g., median)


#######################################################################################


def relative_pose(pose1, pose2):
    """
    Calculate the relative pose of pose2 with respect to pose1.
    
    Parameters:
        pose1 (np.ndarray): 1D array representing pose 1 in the format [x, y, z, yaw, pitch, roll].
        pose2 (np.ndarray): 1D array representing pose 2 in the format [x, y, z, yaw, pitch, roll].
    
    Returns:
        np.ndarray: Relative pose of pose2 with respect to pose1 in the format [x_rel, y_rel, z_rel, yaw_rel, pitch_rel, roll_rel].
    """
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    # Convert pose1 and pose2 to rotation matrices
    r1 = R.from_euler('xyz', pose1[3:], degrees=False).as_matrix()
    r2 = R.from_euler('xyz', pose2[3:], degrees=False).as_matrix()
    
    # Calculate the relative translation
    translation_rel = np.dot(r1.T, pose2[:3] - pose1[:3])
    
    # Calculate the relative rotation
    rotation_rel = np.dot(r1.T, r2)
    relative_euler = R.from_matrix(rotation_rel).as_euler('xyz')
    
    # Combine translation and rotation to form the relative pose
    relative_pose = np.concatenate((translation_rel, relative_euler))
    
    return relative_pose

def transform_to_world_frame(pose1, relative_pose):
    """
    Convert the relative pose back to the world frame.
    
    Parameters:
        pose1 (np.ndarray): 1D array representing pose 1 in the format [x, y, z, yaw, pitch, roll].
        relative_pose (np.ndarray): 1D array representing the relative pose in the format [x_rel, y_rel, z_rel, yaw_rel, pitch_rel, roll_rel].
    
    Returns:
        np.ndarray: Pose of the relative frame in the world frame in the format [x_world, y_world, z_world, yaw_world, pitch_world, roll_world].
    """
    # Convert pose1 to rotation matrix
    r1 = R.from_euler('xyz', pose1[3:], degrees=False).as_matrix()
    
    # Extract relative translation and rotation
    translation_rel = relative_pose[:3]
    rotation_rel = R.from_euler('xyz', relative_pose[3:], degrees=False)
    
    # Transform relative translation to world frame
    translation_world = np.dot(r1, translation_rel) + pose1[:3]
    
    # Transform relative rotation to world frame
    rotation_world = R.from_matrix(np.dot(r1, rotation_rel.as_matrix()))
    euler_world = rotation_world.as_euler('xyz')
    
    # Combine translation and rotation to form the pose in the world frame
    pose_world = np.concatenate((translation_world, euler_world))
    
    return pose_world

def plot_poses(pred, gt, save_path):
    """
    Plot predicted and ground truth poses in 3D along with subplots for roll, pitch, and yaw angles.

    Parameters:
        pred (list): List of predicted poses, where each pose is a list/array of [x, y, z, roll, pitch, yaw].
        gt (list): List of ground truth poses, where each pose is a list/array of [x, y, z, roll, pitch, yaw].
    """

    # Extract x, y, z coordinates and roll, pitch, yaw angles of predicted and ground truth poses
    pred_x, pred_y, pred_z = [], [], []
    pred_roll, pred_pitch, pred_yaw = [], [], []
    for pose in pred:
        pred_x.append(pose[0])
        pred_y.append(pose[1])
        pred_z.append(pose[2])
        pred_roll.append(pose[3])
        pred_pitch.append(pose[4])
        pred_yaw.append(pose[5])

    gt_x, gt_y, gt_z = [], [], []
    gt_roll, gt_pitch, gt_yaw = [], [], []
    for pose in gt:
        gt_x.append(pose[0])
        gt_y.append(pose[1])
        gt_z.append(pose[2])
        gt_roll.append(pose[3])
        gt_pitch.append(pose[4])
        gt_yaw.append(pose[5])

    # Plotting
    fig = plt.figure(figsize=(20, 10))

    # 3D plot
    ax3d = fig.add_subplot(241, projection='3d')
    ax3d.scatter(pred_x, pred_y, pred_z, c='b', marker='o', label='Predicted Poses')
    ax3d.scatter(gt_x, gt_y, gt_z, c='r', marker='^', label='Ground Truth Poses')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Plot')
    ax3d.legend()

    # Top view
    ax_top = fig.add_subplot(242)
    ax_top.scatter(pred_x, pred_y, c='b', marker='o', label='Predicted Poses')
    ax_top.scatter(gt_x, gt_y, c='r', marker='^', label='Ground Truth Poses')
    ax_top.set_xlabel('X')
    ax_top.set_ylabel('Y')
    ax_top.set_title('Top View')

    # Right view
    ax_right = fig.add_subplot(243)
    ax_right.scatter(pred_z, pred_y, c='b', marker='o', label='Predicted Poses')
    ax_right.scatter(gt_z, gt_y, c='r', marker='^', label='Ground Truth Poses')
    ax_right.set_xlabel('Z')
    ax_right.set_ylabel('Y')
    ax_right.set_title('Right View')

    # Front view
    ax_front = fig.add_subplot(244)
    ax_front.scatter(pred_x, pred_z, c='b', marker='o', label='Predicted Poses')
    ax_front.scatter(gt_x, gt_z, c='r', marker='^', label='Ground Truth Poses')
    ax_front.set_xlabel('X')
    ax_front.set_ylabel('Z')
    ax_front.set_title('Front View')

    # Subplot for roll angles
    ax_roll = fig.add_subplot(245)
    ax_roll.plot(pred_roll, label='Predicted Roll')
    ax_roll.plot(gt_roll, label='Ground Truth Roll')
    ax_roll.set_xlabel('Pose Index')
    ax_roll.set_ylabel('Roll Angle (rad)')
    ax_roll.set_title('Roll Angle')
    ax_roll.legend()

    # Subplot for pitch angles
    ax_pitch = fig.add_subplot(246)
    ax_pitch.plot(pred_pitch, label='Predicted Pitch')
    ax_pitch.plot(gt_pitch, label='Ground Truth Pitch')
    ax_pitch.set_xlabel('Pose Index')
    ax_pitch.set_ylabel('Pitch Angle (rad)')
    ax_pitch.set_title('Pitch Angle')
    ax_pitch.legend()

    # Subplot for yaw angles
    ax_yaw = fig.add_subplot(247)
    ax_yaw.plot(pred_yaw, label='Predicted Yaw')
    ax_yaw.plot(gt_yaw, label='Ground Truth Yaw')
    ax_yaw.set_xlabel('Pose Index')
    ax_yaw.set_ylabel('Yaw Angle (rad)')
    ax_yaw.set_title('Yaw Angle')
    ax_yaw.legend()

    # Save plot as image
    plt.tight_layout()
    plt.savefig(save_path)

def plot_loss(train_loss, val_loss, save_path):
    
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.savefig(save_path, format='png', dpi=600 ,bbox_inches='tight')
    plt.close()