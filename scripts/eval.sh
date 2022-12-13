
for experimetn_name in "instant-ngp_test_mono" "instant-ngp_test_stereo" "instant-ngp_test_quad"
do
    for aabb_scale in "2" "4" "8" "16"
    do 
        echo ${experimetn_name} ${aabb_scale}
        python scripts/eval_with_npy.py --pred_path Experiments/Kitti_Experiments/${experimetn_name}/depth_${aabb_scale}/00_0000000000.npy
    done
done

