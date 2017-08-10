from __future__ import division, print_function
import argparse
import sys
import os

import numpy as np

from src.parse_arguments import parse_arguments
from src.tracker import tracker
import src.siamese as siam

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=int, required=True)
parser.add_argument("--y", type=int, required=True)
parser.add_argument("--w", type=int, required=True)
parser.add_argument("--h", type=int, required=True)
parser.add_argument("-s", "--source", help="dir of frame .jpgs",
                    type=str, required=True)


def main(args):
    # avoid printing TF debugging information
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    track_region(
        args.source, args.x + args.w / 2, args.y + args.h / 2, args.w, args.h)


def track_region(source, pos_x, pos_y, target_w, target_h):
    hp, evaluation, run, env, design = parse_arguments()
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1

    # build TF graph once for all
    filename, image, templates_z, scores = siam.build_tracking_graph(
        final_score_sz, design, env)

    frame_names = [os.path.join(source, f)
                   for f in os.listdir(source) if f.endswith(".jpg")]

    frame_names.sort()

    bboxes, speed = tracker(hp, run, design, frame_names,
                            pos_x, pos_y, target_w, target_h, final_score_sz,
                            filename, image, templates_z, scores,
                            start_frame=0)

    print(evaluation.video + ' -- Speed: ' + "%.2f" % speed + ' --')

    return bboxes



if __name__ == '__main__':
    sys.exit(main(args = parser.parse_args()))
