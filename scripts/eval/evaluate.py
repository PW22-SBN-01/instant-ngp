import argparse
import os

import numpy as np
import sys
import math
import cv2
import os

from kitti_iterator.kitti_raw_iterator import KittiRaw
from kitti_iterator.helper import compute_errors

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--kitti_raw_base_path", default=os.path.expanduser("~/Datasets/kitti/raw/"), help="KITTI raw folder")
    parser.add_argument("--date_folder", default="2011_09_26", help="KITTI raw folder")
    parser.add_argument("--sub_folder", default="2011_09_26_drive_0001_sync", help="KITTI raw folder")
    parser.add_argument("--index", default=0, type=int, help="Index")

    parser.add_argument("--pred", default="Experiments/Kitti_Experiments/instant-ngp_test_quad/depth/00_0000000000.npy", help="Path to numpy of predicted depth")
    parser.add_argument("--exposure", default=-5.0, type=float, help="Index")

    parser.add_argument("--plot", action='store_true', help="Enable plotting")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    k_raw = KittiRaw(
        kitti_raw_base_path=args.kitti_raw_base_path,
        date_folder=args.date_folder,
        sub_folder=args.sub_folder,
    )

    depth_map_gt = k_raw[args.index]['depth_image_00']
    depth_map_pred = np.load(args.pred)
    depth_map_pred = depth_map_pred[:,:,:3]

    # depth_map_pred = np.clip(depth_map_pred * 2**args.exposure, 0.0, 1.0)

    # MIN_DEPTH, MAX_DEPTH = 0.083271925, 1.0205717
    # depth_map_pred = (depth_map_pred - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255.0
    depth_map_pred = (depth_map_pred - np.min(depth_map_pred)) / (np.max(depth_map_pred) - np.min(depth_map_pred))

    print('depth_map_pred.shape', depth_map_pred.shape)
    print('depth_map_gt.shape', depth_map_gt.shape)

    print('depth_map_pred.dtype', depth_map_pred.dtype)
    print('depth_map_gt.dtype', depth_map_gt.dtype)

    print('depth_map_pred range', np.min(depth_map_pred), np.max(depth_map_pred))
    print('depth_map_gt range', np.min(depth_map_gt), np.max(depth_map_gt))
    print(compute_errors(depth_map_gt, depth_map_gt))

    print(compute_errors(depth_map_gt, depth_map_pred))

    if args.plot:
        cv2.imshow('depth_map_gt', depth_map_gt)
        cv2.imshow('depth_map_pred', depth_map_pred)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    