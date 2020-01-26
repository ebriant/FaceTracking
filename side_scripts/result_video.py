import cv2
import numpy as np
import config
import os
import argparse


def to_video(dir, name):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(dir, name), fourcc, 30.0, (1440,1440))
    dir_img = os.listdir(dir)

    for img in dir_img:
        print(img)
        frame = cv2.imread(os.path.join(dir, img))
    # for i in range(910,970):
    #     frame = cv2.imread(os.path.join(dir, dir_img[i]))
    #     for j in range(3):
        out.write(frame)
    # for img in dir_img:
    #     frame = cv2.imread(os.path.join(dir, img))
    #     out.write(frame)
    # Release everything if job is finished
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', type=str, default="data/output/171219_2_1HD_blurred/Frames jpg",
                        help='the directory to put in video')
    parser.add_argument('-n', '--name', type=str, default="171219_2_1_frames_3750-4350.mp4",
                        help='the video name')
    args = parser.parse_args()
    to_video(args.dir, args.name)
