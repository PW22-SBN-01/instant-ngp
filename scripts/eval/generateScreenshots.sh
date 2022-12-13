scene=$1
snapshot=$2
screenshot_dir=$3
width=$4
height=$5
python scripts/render.py \
            --mode nerf \
            --scene ${scene} \
            --load_snapshot ${snapshot} \
            --screenshot_transforms ${scene}/transforms.json \
            --width ${width} \
            --height ${height} \
            --exposure -5.0 \
            --screenshot_dir ${screenshot_dir}