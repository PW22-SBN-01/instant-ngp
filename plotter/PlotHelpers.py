from time import sleep
import numpy as np
import open3d as o3d
from skimage.util import view_as_blocks
from skimage import io
from open3d import visualization
import json
from scipy.spatial.transform import Rotation

# Helper
def transformPCD(pcd):
    transformPCD.mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd.rotate(transformPCD.mesh.get_rotation_matrix_from_xyz((np.pi/2,np.pi/2,np.pi/2)))
    pcd.rotate(pcd.get_rotation_matrix_from_xyz((0,-np.pi/2,0)))
    return pcd

# Callbacks
def darkMode(vis: o3d.cuda.pybind.visualization.VisualizerWithKeyCallback):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    return False

def showCoordinateFrame(vis: o3d.cuda.pybind.visualization.VisualizerWithKeyCallback):
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    return False

def render_trajectory(vis: o3d.cuda.pybind.visualization.VisualizerWithKeyCallback):
    camera_trajectory_path = "Experiments/IDDataset/trajectory_o3d.json"
    render_trajectory.index = -1
    render_trajectory.trajectory = o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = render_trajectory
        glb.index += 1
        sleep(0.01)
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index], allow_arbitrary=True)
            pose = glb.trajectory.parameters[glb.index].extrinsic
            print(pose)
            # print(pose)
            # cam = ctr.convert_to_pinhole_camera_parameters()
            # print(cam.intrinsic.intrinsic_matrix)
            # cam.extrinsic = pose        
            # ctr.convert_from_pinhole_camera_parameters(cam)        
        else:
            vis.register_animation_callback(None)
        return False

    vis.register_animation_callback(move_forward)
    return False

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

def plot_trajectory(vis):
    camera_transforms_path = "data/IDDataset/idd1/base_cam.json"
    with open(camera_transforms_path) as camera_transforms_file:
        camera_transforms = json.load(camera_transforms_file)
    trajectory = camera_transforms['path']
    m_keyframes = []

    for cam_frame_id in range(len(trajectory)):
        is_first = len(m_keyframes)==0

        R = np.array(trajectory[cam_frame_id]['R']).reshape(4,)
        T = np.array(trajectory[cam_frame_id]['T']).reshape(3,)
        
        p = np.eye(4,4)
        p[:3,:3] = Rotation.from_quat(R).as_matrix()
        p[:3,3] = T

        slice = trajectory[cam_frame_id]['slice']
        scale = trajectory[cam_frame_id]['scale']
        fov = trajectory[cam_frame_id]['fov']
        if 'dof' in trajectory[cam_frame_id]: dof = trajectory[cam_frame_id]['dof']
        if 'glow_mode' in trajectory[cam_frame_id]: glow_mode = trajectory[cam_frame_id]['glow_mode']
        if 'glow_y_cutoff' in trajectory[cam_frame_id]: glow_y_cutoff = trajectory[cam_frame_id]['glow_y_cutoff']

        if is_first:
            first = p

        pose = p.copy()

        # pose = np.linalg.inv(pose)
        # pose[0:3,2] *= -1 # flip the y and z axis
        # pose[0:3,1] *= -1
        # pose[0:3,0] *= -1
        # pose = pose[[2,1,0,3],:] # swap y and z
        # pose[2,:] *= -1 # flip whole world upside down
        # pose = np.linalg.inv(pose)

        pose[:3,3] *= 255/scale

        extra_move = np.eye(4,4)
        extra_move[:3,3] = np.array([
            0.5,
            -0.5,
            -0.5,
        ])
        pose = np.matmul(pose, extra_move)
        # pose = np.matmul(extra_move, pose)

        # pose[:3,3] *= scale

        m_keyframes.append(pose[:3,3].copy())
        
    m_keyframes = np.array(m_keyframes)
    traj = o3d.geometry.PointCloud()
    traj.points = o3d.utility.Vector3dVector(m_keyframes)
    traj.colors = o3d.utility.Vector3dVector(np.ones_like(m_keyframes) * 255)
    traj = transformPCD(traj)
    vis.add_geometry(traj)
    return False