import bpy
import os
from math import radians
import numpy as np
from mathutils import Vector, Euler
import numpy as np
import matplotlib.pyplot as plt
from IMU_utils import *
from tqdm import tqdm
from trajectories import *
import sys

# from trajectories import clover_trajectory
# Setup file paths
# output_folder = '/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/DATA'  # Make sure to change this to a valid path on your system


def clear_scene():
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.context.view_layer.update()

def spawn_plane(image_path, x, y, z):
    # Create a plane and scale it
    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, location=(0, 0, 0))
    plane = bpy.context.object
    plane.scale = (x, y, z)  # Scaling in X, Y, Z

    # Set up material with texture
    mat = bpy.data.materials.new(name="Plane_Material")
    plane.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    texture_node = nodes.new('ShaderNodeTexImage')
    shader_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    # Load image
    texture_node.image = bpy.data.images.load(image_path)

    # Connect nodes
    links = mat.node_tree.links
    links.new(texture_node.outputs['Color'], shader_node.inputs['Base Color'])
    links.new(shader_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Position nodes for clarity
    texture_node.location = (-300, 0)
    shader_node.location = (0, 0)
    output_node.location = (300, 0)

    # Update the scene
    bpy.context.view_layer.update()

    return plane


def spawn_camera(name, position, rotation):
    # Add a camera to the scene
    bpy.ops.object.camera_add(location=position, rotation=rotation)
    camera = bpy.context.object
    # Set the name and camera properties
    camera.name = name
    camera.data.type = 'PERSP'
    camera.data.lens = 15
    # Make the camera's local axis visible
    camera.show_axis = True
    return camera

def spawn_imu(name, location, rotation):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location, rotation=rotation)
    imu = bpy.context.object
    imu.name = name
    return imu

def spawn_lightsource(name, location, type='AREA', energy=1, size=10):
    # Add sun
    bpy.ops.object.light_add(type='SUN', location=(0,0,10000))
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    return light

def render_frame(frame, output_path):
    bpy.context.scene.start_frameet(frame)
    bpy.context.scene.render.filepath = os.path.join(output_path, f"frame_{frame:04d}")
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # Set output format
    bpy.ops.render.render(write_still=True)


class DataGenerator:
    def __init__(
            self, camera, imu, start_frame, end_frame, 
            fps, time, trajectory_func, pose_file, ref_imu_file, real_imu_file,
            save_path, x_resolution=640, y_resolution=480
    ):
        self.camera = camera
        self.imu = imu
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.image_resolution = (x_resolution, y_resolution)
        self.time = time
        self.trajectory_func = trajectory_func
        self.fps = fps
        self.dt = 1.0 / self.fps
        self.trajectory = []
        self.pose_file = pose_file
        ############# Test ################
        self.ref_imu_file = ref_imu_file
        self.real_imu_file = real_imu_file
        ############# Test ################
        self.data_save_path = save_path

    def compute_angular_velocities(self, q1, q2, dt):
        omega_x = 2/ dt * (q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2])
        omega_y = 2/ dt * (q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1])
        omega_z = 2/ dt * (q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0])
        return np.array([omega_x, omega_y, omega_z])
    
    def render_frame(self, frame, save_path):
        bpy.context.scene.frame_set(frame)
        bpy.context.scene.render.filepath = os.path.join(save_path, f"frame_{frame:04d}")
        bpy.context.scene.render.image_settings.file_format = 'PNG'


    def animate(self):
        scene = bpy.context.scene
        scene.camera = self.camera
        scene.frame_start = self.start_frame
        scene.frame_end = self.end_frame
        scene.render.resolution_x = self.image_resolution[0]
        scene.render.resolution_y = self.image_resolution[1]
        scene.render.resolution_percentage = 100
        bpy.context.scene.render.fps = self.fps

        previous_location = Vector(self.imu.location)
        previous_rotation = Euler(self.imu.rotation_euler, 'XYZ')
        previous_velocity = Vector((0, 0, 0))

        with open(self.pose_file, 'w') as pose_file, open(self.ref_imu_file, 'w') as ref_imu_file, open(self.real_imu_file, 'w') as real_imu_file:
            # for frame, t in enumerate(self.time, start=self.start_frame):
            for frame, t in tqdm(enumerate(self.time, start=self.start_frame), total=len(self.time)):
                x, y, z = self.trajectory_func(t)
                updated_location = Vector((x, y, z))
                pitch = np.radians(30 * np.sin(2 * np.pi * frame / self.end_frame) * np.cos(2 * np.pi * frame / self.end_frame))
                roll = np.radians(30 * np.cos(2 * np.pi * frame / self.end_frame) * np.cos(2 * np.pi * frame / self.end_frame))
                yaw = np.radians(0)
                updated_rotation = Euler((roll, pitch, yaw), 'XYZ')

                velocity = (updated_location - previous_location) / self.dt
                acceleration = (velocity - previous_velocity) / self.dt

                angular_velocity = self.compute_angular_velocities(previous_rotation.to_quaternion(), updated_rotation.to_quaternion(), self.dt)

                previous_location = updated_location
                previous_rotation = updated_rotation
                previous_velocity = velocity

                self.camera.location = (x, y, z) 
                self.camera.rotation_euler = roll, pitch, yaw

                self.camera.keyframe_insert(data_path="location", frame=frame)
                self.camera.keyframe_insert(data_path="rotation_euler", frame=frame)

                self.trajectory.append([x, y, z])

                reference_acceleration = [acceleration[0], acceleration[1], acceleration[2]]
                real_acceleration = run_acc_demo(np.array([reference_acceleration])).squeeze()

                reference_gyro = [angular_velocity[0], angular_velocity[1], angular_velocity[2]]
                real_gyro = run_gyro_demo(np.array([reference_gyro])).squeeze()

                if frame < 3: 
                    continue
                else:
                    # camera Pose  
                    pose_file.write(f'Frame {frame/self.end_frame:.4f}: Position ( {x:.6f} , {y:.6f} , {z:.6f} ) , Rotation ( {roll:.6f} , {pitch:.6f} , {yaw:.6f} )\n')

                    # IMU data
                    ref_imu_file.write(f'Frame {frame/self.end_frame:.4f} {acceleration[0]:.6f} {acceleration[1]:.6f} {acceleration[2]:.6f} {angular_velocity[0]:.6f} {angular_velocity[1]:.6f} {angular_velocity[2]:.6f}\n')
                    real_imu_file.write(f'Frame {frame/self.end_frame:.4f} {real_acceleration[0]:.6f} {real_acceleration[1]:.6f} {real_acceleration[2]:.6f} {real_gyro[0]:.6f} {real_gyro[1]:.6f} {real_gyro[2]:.6f}\n')

                self.render_frame(frame, self.data_save_path)
                bpy.ops.render.render(write_still=True)
                bpy.context.view_layer.update()

        return self.trajectory



def main(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder, 'images'))
    # Clear the scene of any objects
    clear_scene()

    # Texture path
    texture_path = "/home/megatron/Workspace/WPI/Sem2/RBE549-Computer_Vision/Projects/P4/Phase2/texture_image/plane_render.jpg"
    # Camera Parameters
    camera_rotation = (radians(0), 0, 0)
    camera_position = (0, 0, 10)

    # Spawn Objects in the scene
    # 1. Spawn a plane
    plane = spawn_plane(texture_path, x=600, y=600, z=0)
    # 2. Spawn a new camera
    camera = spawn_camera("CameraObject", camera_position, camera_rotation)
    # 3. Spawn a light source
    light = spawn_lightsource("LightSource", (0, 0, 20), 'AREA', 1, 10)
    # 4. Spawn an IMU
    imu = spawn_imu("IMU", camera_position, camera_rotation)

    start_frame = 1      # frame start
    end_frame = 5000     # frame end

    time = np.linspace(start_frame, 10,end_frame - start_frame + 1)

    # Generate Data
    function = archimedean_spiral_flat_rotated
    data_gen = DataGenerator(
        camera, imu, start_frame, end_frame, 60, time, function, 
        os.path.join(output_folder, 'camera_poses.txt'), os.path.join(output_folder, 'ref_imu_data.txt'), 
        os.path.join(output_folder, 'real_imu_data.txt'), os.path.join(output_folder, 'images')
    )
    trajectory = data_gen.animate()


    # plot the trajectory x y z
    # import matplotlib.pyplot as plt
    x_waypoints = [x[0] for x in trajectory]
    y_waypoints = [x[1] for x in trajectory]
    z_waypoints = [x[2] for x in trajectory]

    # Write the trajectory to a file
    with open(os.path.join(output_folder, 'trajectory_test.txt'), 'w') as traj_file:
        for i, (x, y, z) in enumerate(trajectory):
            traj_file.write(f"{x} {y} {z}\n")


    # Plot figure-of-eight trajectory
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_waypoints, y_waypoints, z_waypoints, label=function.__name__)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    #save reference figure
    fig.savefig(os.path.join(output_folder, 'trajectory.png'), bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    output_folder = sys.argv[1] if len(sys.argv) > 1 else '/default/output/path'
    main(output_folder)