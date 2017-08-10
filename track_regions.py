from __future__ import division, print_function
import argparse
import sys
import os

import numpy as np
import pandas

from track import track_region

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
    for i, r in enumerate(pandas.read_csv(args.csv).iterrows()):
        outname = os.path.join(args.dest, args.csv + str(i) + ".csv")
        file_name = os.path.join(args.dir, r["filename"])

        x = r["xmin"]
        y = r["ymin"]
        h = r["ymax"] - r["ymin"]
        w = r["xmax"] - r["xmin"]

        bboxes = track_region(file_name, x +  w / 2, y + h / 2, w, h)

        df = pandas.DataFrame(bboxes, columns=("x", "y", "width", "height"))
        df.write_csv(outname)


if __name__ == '__main__':
    sys.exit(main(args = parser.parse_args()))
