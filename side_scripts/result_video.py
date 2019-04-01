import cv2
import numpy as np
import config
import os
import argparse


def to_video(dir, name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(dir, name), fourcc, 30.0, (640,640))
    dir_img = os.listdir(dir)
    for img in dir_img:
        frame = cv2.imread(os.path.join(dir, img))
        out.write(frame)
    # Release everything if job is finished
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', type=str, default="data/output/171214_1_old",
                        help='the directory to put in video')
    parser.add_argument('-n', '--name', type=str, default="video.mp4",
                        help='the video name')

    args = parser.parse_args()
    to_video(args.dir, args.name)
