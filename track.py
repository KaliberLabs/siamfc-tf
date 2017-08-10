from __future__ import division, print_function
import argparse
import sys
import os

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
    tracky = Tracker()
    tracky.track_region(args.source, args.x, args.y, args.w, args.h)


class Tracker:
    def __init__(self):
        self.hyperparameters, self.evaluation, self.run, self.env, self.design = \
            parse_arguments()

        self.final_score_sz = self.hyperparameters.response_up * (
            self.design.score_sz - 1) + 1

        # build TF graph once for all
        self.filename, self.image, self.templates_z, self.scores = \
            siam.build_tracking_graph(self.final_score_sz, self.design, self.env)

    def track_region(self, source, x, y, w, h, start_frame=0):
        frame_names = [os.path.join(source, f)
                       for f in os.listdir(source) if f.endswith(".jpg")]

        assert len(frame_names) > 0, ".jpg files found in " + source

        frame_names.sort()

        center_x = x + w / 2
        center_y = y + h / 2

        bboxes, speed = tracker(
            hp=self.hyperparameters,
            run=self.run,
            design=self.design,
            frame_name_list=frame_names,
            pos_x=center_x,
            pos_y=center_y,
            target_w=w,
            target_h=h,
            final_score_sz=self.final_score_sz,
            filename=self.filename,
            image=self.image,
            templates_z=self.templates_z,
            scores=self.scores,
            start_frame=start_frame)

        print(self.evaluation.video + ' -- Speed: ' + "%.2f" % speed + ' --')

        return bboxes


if __name__ == '__main__':
    sys.exit(main(args = parser.parse_args()))
