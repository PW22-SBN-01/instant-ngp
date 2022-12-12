import open3d as o3d
import argparse
import os
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot out the camera trajectory from transforms.json")

    parser.add_argument("--transforms_paths", default=[os.path.expanduser("transforms.json"), ], nargs='+', help="transforms.json path")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = np.array([0,0,0])

    for transforms_path in args.transforms_paths:
        with open(transforms_path) as camera_transforms_file:
            camera_transforms = json.load(camera_transforms_file)

        points = []

        for frame in camera_transforms['frames']:
            transform_matrix = np.array(frame['transform_matrix'])
            points.append(transform_matrix[:3,3])

        points = np.array(points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()