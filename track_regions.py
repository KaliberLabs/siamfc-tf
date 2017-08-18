#!/usr/bin/env python
"""
This code was originally written for Py2. If you get a weird error, try with Py2.
"""
from __future__ import division, print_function
import argparse
import sys
import csv
import os

from PIL import Image
import pandas

from track import Tracker

parser = argparse.ArgumentParser(
    description="Take the csv from person detection and"
                "output a directory of csvs of positions across time")
parser.add_argument("-i", "--input", type=str, required=True)
parser.add_argument("--source", help="dir of frame .jpgs", type=str,
                    required=True, dest="source")
parser.add_argument("-o", "--output-dir", help="destinations of output csvs",
                    type=str, required=True, dest="output_dir")
parser.add_argument("--relative", dest="relative", action="store_true",
                    default=False,
                    help="Output relative coordinates")
parser.add_argument("--start-frame", default=0, help="frame to begin tracking at")
parser.add_argument("--tensor-flow-log-level", default=3, type=int)


def main(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tensor_flow_log_level)

    if not args.input.endswith(".csv"):
        raise ValueError("Input file must be a .csv. %s given" % args.input)

    with open(args.input) as csv_file:
        tracker = Tracker()
        for i, row in enumerate(csv.DictReader(csv_file)):
            out_path = os.path.join(args.output_dir, "%i.csv" % i)

            start_frame = os.path.join(args.source, row["filename"])
            img_width, img_height = Image.open(start_frame).size

            x = int(float(row["xmin"]) * img_width)
            y = int(float(row["ymin"]) * img_height)
            h = int((float(row["ymax"]) - y) * img_height)
            w = int((float(row["xmax"]) - x) * img_width)

            print("Processing", args.start_frame, "(x y w h)", x, y, w, h)

            bboxes, frames = tracker.track_region(args.source, x, y, w, h,
                                                  start_frame=args.start_frame)

            if args.relative:
                # convert back to relative coordinates
                bboxes[:, (0, 2)] /= img_width
                bboxes[:, (1, 3)] /= img_height
                bboxes.round(6, bboxes) # Round to 6 decimal points in-pace

            df = pandas.DataFrame(bboxes, columns=("x", "y", "width", "height"))
            df["filename"] = frames

            df.to_csv(out_path, index=True, index_label="frame")


if __name__ == "__main__":
    sys.exit(main(args=parser.parse_args()))
