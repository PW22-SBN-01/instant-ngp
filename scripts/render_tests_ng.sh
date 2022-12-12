python scripts/run.py \
    --mode nerf \
    --scene data/rr1 \
    --load_snapshot data/rr1/base.msgpack \
    --video_camera_path data/abc/base_cam.json \
    --video_n_seconds 5 \
    --video_fps 30 \
    --width 1920 \
    --height 1080


python scripts/run.py \
    --mode nerf \
    --scene data/KittiDataset_2/images \
    --load_snapshot data/KittiDataset_2/images/base.msgpack \
    --video_camera_path data/KittiDataset_2/images/base_cam.json \
    --video_n_seconds 5 \
    --video_fps 30 \
    --width 1920 \
    --height 1080 --gui


python scripts/run.py \
    --mode nerf \
    --scene data/KittiDataset_2/images \
    --load_snapshot data/KittiDataset_2/images/base.msgpack \
    --video_camera_path data/KittiDataset_2/images/base_cam.json \
    --video_n_seconds 1 \
    --video_fps 10 \
    --width 1920 \
    --height 1080 --gui


python scripts/run.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_mono/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_mono/base_2.msgpack \
    --video_camera_path Experiments/Kitti_Experiments/instant-ngp_test_mono/base_cam.json \
    --video_n_seconds 1 \
    --video_fps 10 \
    --width 1920 \
    --height 1080 --gui


python scripts/run.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_mono/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_mono/base_16.msgpack \
    --video_camera_path Experiments/Kitti_Experiments/instant-ngp_test_mono/base_cam.json \
    --video_n_seconds 1 \
    --video_fps 10 \
    --width 1920 \
    --height 1080 --gui

python scripts/run.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_quad/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_quad/base_16.msgpack \
    --video_camera_path Experiments/Kitti_Experiments/instant-ngp_test_quad/base_cam.json \
    --video_n_seconds 10 \
    --video_fps 10 \
    --width 1920 \
    --height 1080 --gui

python scripts/run.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_quad/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_quad/base_16.msgpack \
    --video_camera_path Experiments/Kitti_Experiments/instant-ngp_test_quad/base_cam.json \
    --video_n_seconds 5 \
    --video_fps 30 \
    --width 1920 \
    --exposure -5.0 \
    --height 1080 --gui


python scripts/render.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_quad/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_quad/base_16.msgpack \
    --screenshot_transforms Experiments/Kitti_Experiments/instant-ngp_test_quad/transforms.json \
    --width 1242 \
    --height 375 \
    --exposure -5.0 \
    --screenshot_dir Experiments/Kitti_Experiments/instant-ngp_test_quad/depth


python scripts/render.py \
    --mode nerf \
    --scene Experiments/Kitti_Experiments/instant-ngp_test_quad/ \
    --load_snapshot Experiments/Kitti_Experiments/instant-ngp_test_mono/base_16.msgpack \
    --screenshot_transforms transforms.json \
    --width 1242 \
    --height 375 \
    --exposure -5.0 \
    --screenshot_dir screenshot_dir/ --gui