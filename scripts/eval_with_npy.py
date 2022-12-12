"""

"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import fnmatch
import cv2
import numpy as np
from kitti_iterator.kitti_raw_iterator import KittiRaw


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='Evaluation script', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png',  default="Experiments/Kitti_Experiments/instant-ngp_test_quad/depth/00_0000000000.npy")

parser.add_argument("--kitti_raw_base_path", default=os.path.expanduser("~/Datasets/kitti/raw/"), help="KITTI raw folder")
parser.add_argument("--date_folder", default="2011_09_26", help="KITTI raw folder")
parser.add_argument("--sub_folder", default="2011_09_26_drive_0001_sync", help="KITTI raw folder")
parser.add_argument("--index", default=0, type=int, help="Index")

parser.add_argument('--dataset',             type=str,   help='dataset to test on, nyu or kitti', default='nyu')
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')

args = parser.parse_args()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def test():
    global gt_depths, missing_ids, pred_filenames
    gt_depths = []
    missing_ids = set()
    pred_filenames = []

    pred_depths = []
    gt_depths = []

    pred_filenames = [args.pred_path]

    num_test_samples = len(pred_filenames)

    k_raw = KittiRaw(
        kitti_raw_base_path=args.kitti_raw_base_path,
        date_folder=args.date_folder,
        sub_folder=args.sub_folder,
    )

    depth_map_gt = k_raw[args.index]['depth_image_00']
    depth_map_pred = np.load(args.pred_path)
    depth_map_pred = depth_map_pred[:,:,:3]

    # depth = depth.astype(np.float32) / 256.0

    pred_depths = [depth_map_pred, ]
    gt_depths = [depth_map_gt]

    eval(pred_depths)


def eval(pred_depths):

    num_samples = len(pred_depths)
    pred_depths_valid = []

    i = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths_valid[i]

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


def main():
    test()


if __name__ == '__main__':
    main()


