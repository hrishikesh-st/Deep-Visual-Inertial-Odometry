"""
Class for plotting a quadrotor

Inspired from code written by Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from natsort import natsorted

class Quadrotor():
    def __init__(self, gt, pred, size=0.25, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T
        self.p5 = np.array([0, 0, size/4, 1]).T

        self.gt_x_data = []
        self.gt_y_data = []
        self.gt_z_data = []

        self.pred_x_data = []
        self.pred_y_data = []
        self.pred_z_data = []

        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(gt, pred, 0)

    def update_pose(self, gt, pred, i):
        self.gt_x = gt[0]
        self.gt_y = gt[1]
        self.gt_z = gt[2]
        self.gt_roll = gt[3]
        self.gt_pitch = gt[4]
        self.gt_yaw = gt[5]

        self.pred_x = pred[0]
        self.pred_y = pred[1]
        self.pred_z = pred[2]
        self.pred_roll = pred[3]
        self.pred_pitch = pred[4]
        self.pred_yaw = pred[5]

        self.gt_x_data.append(gt[0])
        self.gt_y_data.append(gt[1])
        self.gt_z_data.append(gt[2])

        self.pred_x_data.append(pred[0])
        self.pred_y_data.append(pred[1])
        self.pred_z_data.append(pred[2])

        if self.show_animation:
            self.plot(i)

    def transformation_matrix(self):
        gt_x = self.gt_x
        gt_y = self.gt_y
        gt_z = self.gt_z
        gt_roll = self.gt_roll
        gt_pitch = self.gt_pitch
        gt_yaw = self.gt_yaw

        pred_x = self.pred_x
        pred_y = self.pred_y
        pred_z = self.pred_z
        pred_roll = self.pred_roll
        pred_pitch = self.pred_pitch
        pred_yaw = self.pred_yaw

        T_gt = np.array(
            [[cos(gt_yaw) * cos(gt_pitch), -sin(gt_yaw) * cos(gt_roll) + cos(gt_yaw) * sin(gt_pitch) * sin(gt_roll), sin(gt_yaw) * sin(gt_roll) + cos(gt_yaw) * sin(gt_pitch) * cos(gt_roll), gt_x],
             [sin(gt_yaw) * cos(gt_pitch), cos(gt_yaw) * cos(gt_roll) + sin(gt_yaw) * sin(gt_pitch)
              * sin(gt_roll), -cos(gt_yaw) * sin(gt_roll) + sin(gt_yaw) * sin(gt_pitch) * cos(gt_roll), gt_y],
             [-sin(gt_pitch), cos(gt_pitch) * sin(gt_roll), cos(gt_pitch) * cos(gt_roll), gt_z]
             ])

        T_pred = np.array(
            [[cos(pred_yaw) * cos(pred_pitch), -sin(pred_yaw) * cos(pred_roll) + cos(pred_yaw) * sin(pred_pitch) * sin(pred_roll), sin(pred_yaw) * sin(pred_roll) + cos(pred_yaw) * sin(pred_pitch) * cos(pred_roll), pred_x],
             [sin(pred_yaw) * cos(pred_pitch), cos(pred_yaw) * cos(pred_roll) + sin(pred_yaw) * sin(pred_pitch)
              * sin(pred_roll), -cos(pred_yaw) * sin(pred_roll) + sin(pred_yaw) * sin(pred_pitch) * cos(pred_roll), pred_y],
             [-sin(pred_pitch), cos(pred_pitch) * sin(pred_roll), cos(pred_pitch) * cos(pred_roll), pred_z]
             ])

        return T_gt, T_pred

    def plot(self, i):  # pragma: no cover
        T_gt, T_pred = self.transformation_matrix()

        gt_p1_t = np.matmul(T_gt, self.p1)
        gt_p2_t = np.matmul(T_gt, self.p2)
        gt_p3_t = np.matmul(T_gt, self.p3)
        gt_p4_t = np.matmul(T_gt, self.p4)
        gt_p5_t = np.matmul(T_gt, self.p5)
        _center = np.array([self.gt_x, self.gt_y, self.gt_z])

        plt.cla()

        self.ax.plot([gt_p1_t[0], gt_p2_t[0], gt_p3_t[0], gt_p4_t[0]],
                     [gt_p1_t[1], gt_p2_t[1], gt_p3_t[1], gt_p4_t[1]],
                     [gt_p1_t[2], gt_p2_t[2], gt_p3_t[2], gt_p4_t[2]], 'ko', markersize=8, alpha=0.5)

        self.ax.plot([gt_p1_t[0], gt_p2_t[0]], [gt_p1_t[1], gt_p2_t[1]],
                     [gt_p1_t[2], gt_p2_t[2]], 'r-', linewidth=2, alpha=0.5)
        self.ax.plot([gt_p3_t[0], gt_p4_t[0]], [gt_p3_t[1], gt_p4_t[1]],
                     [gt_p3_t[2], gt_p4_t[2]], 'g-', linewidth=2, alpha=0.5)
        self.ax.plot([_center[0], gt_p5_t[0]], [_center[1], gt_p5_t[1]],
                     [_center[2], gt_p5_t[2]], 'b-', linewidth=2, alpha=0.5)

        self.ax.plot(self.gt_x_data, self.gt_y_data, self.gt_z_data, 'c:', linewidth=2, alpha=0.5, label='Groundtruth')

        pred_p1_t = np.matmul(T_pred, self.p1)
        pred_p2_t = np.matmul(T_pred, self.p2)
        pred_p3_t = np.matmul(T_pred, self.p3)
        pred_p4_t = np.matmul(T_pred, self.p4)
        pred_p5_t = np.matmul(T_pred, self.p5)
        _center = np.array([self.pred_x, self.pred_y, self.pred_z])

        # plt.cla()

        self.ax.plot([pred_p1_t[0], pred_p2_t[0], pred_p3_t[0], pred_p4_t[0]],
                     [pred_p1_t[1], pred_p2_t[1], pred_p3_t[1], pred_p4_t[1]],
                     [pred_p1_t[2], pred_p2_t[2], pred_p3_t[2], pred_p4_t[2]], 'ko', markersize=8)

        self.ax.plot([pred_p1_t[0], pred_p2_t[0]], [pred_p1_t[1], pred_p2_t[1]],
                     [pred_p1_t[2], pred_p2_t[2]], 'r-', linewidth=2)
        self.ax.plot([pred_p3_t[0], pred_p4_t[0]], [pred_p3_t[1], pred_p4_t[1]],
                     [pred_p3_t[2], pred_p4_t[2]], 'g-', linewidth=2)
        self.ax.plot([_center[0], pred_p5_t[0]], [_center[1], pred_p5_t[1]],
                     [_center[2], pred_p5_t[2]], 'b-', linewidth=2)

        self.ax.plot(self.pred_x_data, self.pred_y_data, self.pred_z_data, 'm:', linewidth=2, label='Predicted')

        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        self.ax.set_zlim(-10, 40)

        # self.ax.view_init(60, 60)
        self.ax.legend()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.title('Quadrotor Simulation - Bent Spiral Trajectory')
        plt.tight_layout()
        plt.savefig('/home/tejasrane/wpi_courses/RBE549_Computer_Vision/P4/scratchpad/trajs/bent_spiral2/vio/drone_'+str(i)+'.png', dpi=600, bbox_inches='tight')
        # plt.pause(0.001)

if __name__ == '__main__':
    with open('/home/tejasrane/wpi_courses/RBE549_Computer_Vision/P4/scratchpad/trajs/bent_spiral2/vio/gt_pose.txt', 'r') as file:
        gt_lines = file.readlines()

    with open('/home/tejasrane/wpi_courses/RBE549_Computer_Vision/P4/scratchpad/trajs/bent_spiral2/vio/pred_pose.txt', 'r') as file:
        pred_lines = file.readlines()

    _gt = gt_lines[0].split()
    _pred = pred_lines[0].split()
    q = Quadrotor([float(_gt[0]), float(_gt[1]), float(_gt[2]), float(_gt[3]), float(_gt[4]), float(_gt[5])],
                  [float(_pred[0]), float(_pred[1]), float(_pred[2]), float(_pred[3]), float(_pred[4]), float(_pred[5])], size=15)

    for i in range(1, len(gt_lines), 5):
        _gt = gt_lines[i].split()
        _pred = pred_lines[i].split()
        q.update_pose([float(_gt[0]), float(_gt[1]), float(_gt[2]), float(_gt[3]), float(_gt[4]), float(_gt[5])],
                      [float(_pred[0]), float(_pred[1]), float(_pred[2]), float(_pred[3]), float(_pred[4]), float(_pred[5])], i)

    input_dir = '/home/tejasrane/wpi_courses/RBE549_Computer_Vision/P4/scratchpad/trajs/bent_spiral2/vio/'

    # Output video file
    output_video = "/home/tejasrane/wpi_courses/RBE549_Computer_Vision/P4/scratchpad/trajs/bent_spiral2/vio/output_video.mp4"

    # Frame rate (fps) of the output video
    frame_rate = 15

    # Get the list of image filenames in the input directory
    image_files = sorted(os.listdir(input_dir))
    image_files = natsorted([f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))])

    # Get the first image to extract dimensions
    first_image = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width, _ = first_image.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec for the output video
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Loop over each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

    print("Video created successfully!")
