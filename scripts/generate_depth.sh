
for experimetn_name in "instant-ngp_test_mono" "instant-ngp_test_stereo" "instant-ngp_test_quad"
do
    for aabb_scale in "2" "4" "8" "16"
    do 
        python scripts/render.py \
            --mode nerf \
            --scene Experiments/Kitti_Experiments/${experimetn_name}/ \
            --load_snapshot Experiments/Kitti_Experiments/${experimetn_name}/base_${aabb_scale}.msgpack \
            --screenshot_transforms Experiments/Kitti_Experiments/${experimetn_name}/transforms.json \
            --width 1242 \
            --height 375 \
            --exposure -5.0 \
            --screenshot_dir Experiments/Kitti_Experiments/${experimetn_name}/depth_${aabb_scale}
    done
done