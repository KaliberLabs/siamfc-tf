from __future__ import division, print_function
import argparse
import sys
import csv
import os

import tensorflow as tf
from PIL import Image
import pandas

from track import Tracker

parser = argparse.ArgumentParser(
    description="Take the csv output from detection and containing photos then "
                "output a csv to stdout of positions across time")
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--source", help="dir of frame .jpgs", type=str,
                    required=True)
parser.add_argument("--dest", help="destinations of output csvs", type=str,
                    required=True)

# avoid printing TF debugging information
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):
    with open(args.csv) as csv_file:
        tracker = Tracker()
        for i, r in enumerate(csv.DictReader(csv_file)):
            out_filename = args.csv.replace(".csv", ".%i.csv" % i)
            outname = os.path.join(args.dest, out_filename)

            start_frame = os.path.join(args.source, r["filename"])
            img_width, img_height = Image.open(start_frame).size

            x = int(float(r["xmax"]) * img_width)
            y = int(float(r["ymax"]) * img_height)
            h = int(y - float(r["ymin"]) * img_height)
            w = int(x - float(r["xmin"]) * img_width)

            #@TODO: FILTER TO START ON SPECFIC FRAMES
            print("INITIALIZING on", start_frame, args.source,
                  "(xywh)", x, y, w, h)

            bboxes = tracker.track_region(args.source, x, y, w, h)

            # convert back to relative coordinates
            bboxes[:, (0, 2)] /= img_width
            bboxes[:, (1, 3)] /= img_height
            bboxes.round(6, bboxes) # Round to 6 decimal points in-pace

            df = pandas.DataFrame(bboxes, columns=("x", "y", "width", "height"))
            df.to_csv(outname, index=False)


if __name__ == '__main__':
    sys.exit(main(args = parser.parse_args()))
