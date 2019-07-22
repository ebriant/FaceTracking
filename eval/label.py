import matplotlib.image as mpimg
import numpy as np
import argparse
import os
import config
import cv2
import utils
import visualization
SCALING = 1.2


class Labeler:
    def __init__(self, names_list, overwrite):
        self.visualizer = visualization.VisualizerOpencv()
        self.s_frames = utils.load_seq_video()
        self.overwrite = overwrite
        self.names_list = names_list
        _, video_name = os.path.split(config.video_path)
        self.dump_file = os.path.join(config.label_dir, "{}_test.txt".format(video_name[:-4]))
        self.data = {}
        if os.path.isfile(self.dump_file):
            with open(self.dump_file, "r") as f:
                self.data = eval(f.read())

    def get_data(self):
        for idx in range(0, len(self.s_frames), config.label_frame_step):
            if self.overwrite or not self.data_exists(idx):
                img = mpimg.imread(self.s_frames[idx])
                img = np.array(img)
                img = cv2.resize(img, None, fx=SCALING, fy=SCALING)
                frame_data = None
                while frame_data is None or len(frame_data) != len(self.names_list):
                    self.visualizer.prepare_img(img, idx)
                    frame_data = self.visualizer.ask_ground_truth()
                frame_data = [(int(a[0]//SCALING), int(a[1]//SCALING)) for a in frame_data]
                self.data[idx] = {self.names_list[i]: point for i, point in enumerate(frame_data)}
                self.save_data()

    def get_bboxes(self):
        last_data = []
        for idx in range(0, len(self.s_frames), config.label_frame_step):
            if self.overwrite or not self.data_exists(idx):
                img = mpimg.imread(self.s_frames[idx])
                img = np.array(img)
                img = cv2.resize(img, None, fx=SCALING, fy=SCALING)
                frame_data = None
                while frame_data is None or len(frame_data) != len(self.names_list):
                    # if idx > 0:
                    #
                    #     continue

                    self.visualizer.prepare_img(img, idx)

                    frame_data = self.visualizer.ask_bboxes()

                frame_data = [[int(i//SCALING) for i in a] for a in frame_data]
                last_data = [i for i in frame_data]
                self.data[idx] = {self.names_list[i]: bbox for i, bbox in enumerate(frame_data)}

                self.save_data()

    def data_exists(self, index):
        return index in self.data

    def save_data(self):
        with open(self.dump_file, "w+") as f:
            f.write(str(self.data))

    def sort_data(self):
        self.data = {key: self.data[key] for key in sorted(self.data)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--names', type=list)
    parser.add_argument('-o', '--overwrite', action='store_true', default=False)
    parser.add_argument('-m', '--mode', type=str, choices=['point', 'bbox'], help="mode of operation")
    args = parser.parse_args()

    labeler = Labeler(args.names, args.overwrite)
    if args.mode == "point":
        labeler.get_data()
    elif args.mode == "bbox":
        labeler.get_bboxes()

    # labeler.sort_data()
    # labeler.save_data()
    # labeler.get_data()
    # labeler.get_bboxes()
