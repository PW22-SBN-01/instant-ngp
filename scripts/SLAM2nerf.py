#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

from kitti_iterator.kitti_raw_iterator import KittiRaw

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--kitti_raw_base_path", default=os.path.expanduser("~/Datasets/kitti/raw/"), help="KITTI raw folder")
    parser.add_argument("--date_folder", default="2011_09_26", help="KITTI raw folder")
    parser.add_argument("--sub_folder", default="2011_09_26_drive_0001_sync", help="KITTI raw folder")

    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")

    parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
    parser.add_argument("--out", default="Experiments/Kitti_Experiments/instant-ngp_test_quad_copy/transforms.json", help="output path")
    parser.add_argument("--vocab_path", default="", help="vocabulary tree path")
    args = parser.parse_args()
    return args

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = "\"" + args.images + "\""
    video =  "\"" + args.video_in + "\""
    fps = float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    try:
        # Passing Images' Path Without Double Quotes
        shutil.rmtree(args.images)
    except:
        pass
    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
    args = parse_args()

    k_raw = KittiRaw(
        kitti_raw_base_path=args.kitti_raw_base_path,
        date_folder=args.date_folder,
        sub_folder=args.sub_folder,
        compute_trajectory=True,
        invalidate_cache=False,
        scale_factor=1.0,
        num_features=5000
    )
    
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    TEXT_FOLDER = args.text
    OUT_PATH = args.out
    print(f"outputting to {OUT_PATH}...")

    w = float(k_raw.width)
    h = float(k_raw.height)
    fl_x = float(k_raw.K_00[0,0])
    fl_y = float(k_raw.K_00[1,1])
    k1 = float(k_raw.D_00[0,0])
    k2 = float(k_raw.D_00[0,1])
    p1 = float(k_raw.D_00[0,2])
    p2 = float(k_raw.D_00[0,3])
    cx = w / 2
    cy = h / 2

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi
         
    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")
    # print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tk={k1,k2} p={p1,p2} ")

    # with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
    i = 0
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
        "frames": [],
    }

    # TSCALE = 2.75
    TSCALE = 3.0

    c0 = np.concatenate([np.concatenate([k_raw.R_00, k_raw.T_00*TSCALE], 1), bottom], 0)
    c1 = np.concatenate([np.concatenate([k_raw.R_01, k_raw.T_01*TSCALE], 1), bottom], 0)
    c2 = np.concatenate([np.concatenate([k_raw.R_02, k_raw.T_02*TSCALE], 1), bottom], 0)
    c3 = np.concatenate([np.concatenate([k_raw.R_03, k_raw.T_03*TSCALE], 1), bottom], 0)

    BASE_FOLDER = os.path.dirname(os.path.abspath(OUT_PATH))
    DATA_FOLDER = os.path.join(BASE_FOLDER,'data')
    os.makedirs(BASE_FOLDER, exist_ok=True)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    up = np.zeros(3)
    for index, data_frame in enumerate(k_raw):
        if index<2:
            continue

        trajectory = k_raw.trajectory.iloc[index]

        print(index, trajectory)
        image_id = k_raw.img_list[index]

        image_00_raw = data_frame['image_00_raw']
        image_01_raw = data_frame['image_01_raw']
        image_02_raw = data_frame['image_02_raw']
        image_03_raw = data_frame['image_03_raw']

        image_00_raw = cv2.cvtColor(image_00_raw, cv2.COLOR_BGR2GRAY)
        image_01_raw = cv2.cvtColor(image_01_raw, cv2.COLOR_BGR2GRAY)
        image_02_raw = cv2.cvtColor(image_02_raw, cv2.COLOR_BGR2GRAY)
        image_03_raw = cv2.cvtColor(image_03_raw, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(DATA_FOLDER, str(image_id) + '_image_00_raw.png'), image_00_raw)
        cv2.imwrite(os.path.join(DATA_FOLDER, str(image_id) + '_image_01_raw.png'), image_01_raw)
        cv2.imwrite(os.path.join(DATA_FOLDER, str(image_id) + '_image_02_raw.png'), image_02_raw)
        cv2.imwrite(os.path.join(DATA_FOLDER, str(image_id) + '_image_03_raw.png'), image_03_raw)
        
        name = os.path.join(k_raw.image_00_path, 'data', image_id + ".png")
        name_rel = os.path.relpath(name, BASE_FOLDER)

        print(name)

        b=sharpness(name)
        print(name_rel, "sharpness=",b)
        tvec = np.array(tuple(map(float, [trajectory['x'], trajectory['y'], trajectory['z']])))
        R = trajectory['rot']
        t = tvec.reshape([3,1])
        m_ref = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

        # m0 = np.matmul(m_ref, c0)
        # m1 = np.matmul(m_ref, c1)
        # m2 = np.matmul(m_ref, c2)
        # m3 = np.matmul(m_ref, c3)

        m0 = np.matmul(c0, m_ref)
        m1 = np.matmul(c1, m_ref)
        m2 = np.matmul(c2, m_ref)
        m3 = np.matmul(c3, m_ref)

        for cam_name, m in [
            ('_image_00_raw', m0), 
            ('_image_01_raw', m1), 
            ('_image_02_raw', m2), 
            ('_image_03_raw', m3), ]:
            c2w = np.linalg.inv(m)
            if not args.keep_colmap_coords:
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                c2w = c2w[[1,0,2,3],:] # swap y and z
                c2w[2,:] *= -1 # flip whole world upside down

                up += c2w[0:3,1]

            name_cam = os.path.join(DATA_FOLDER, str(image_id) + cam_name + '.png')
            name_rel = os.path.relpath(name_cam, BASE_FOLDER)
            print(name_rel)
            frame={"file_path":name_rel,"sharpness":b,"transform_matrix": c2w}
            out["frames"].append(frame)

    nframes = len(out["frames"])

    if args.keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)
