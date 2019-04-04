import utils
import visualization
import matplotlib.image as mpimg
import numpy as np
import argparse
import os
import config

RATE = 30


class Labeler:
    def __init__(self, names_list, dump_file, overwrite):
        self.visualizer = visualization.VisualizerOpencv()
        self.s_frames = utils.load_seq_video()
        self.overwrite = overwrite
        self.names_list = names_list
        _, video_name = os.path.split(config.video_path)
        label_dir_path = os.path.join(config.label_dir, video_name[:-4])
        if not os.path.exists(label_dir_path):
            os.mkdir(label_dir_path)
        self.dump_file = os.path.join(label_dir_path, dump_file)
        print("aaaaa", os.path.isfile(self.dump_file))
        if os.path.isfile(self.dump_file):
            with open(self.dump_file, "r") as f:
                self.data = eval(f.read())
        else:
            self.data = {}
        # except FileNotFoundError:
        #     self.data = {}

    def get_data(self):
        for idx in range(0, len(self.s_frames), RATE):
            if self.overwrite or not os.path.isfile(self.dump_file) or not self.data_exists(idx):
                img = mpimg.imread(self.s_frames[idx])
                img = np.array(img)
                self.visualizer.prepare_img(img, idx)
                frame_data = self.visualizer.ask_ground_truth(pt_nb=len(self.names_list))
                self.data[idx] = {self.names_list[i]: point for i, point in enumerate(frame_data)}
                self.save_data()

    def data_exists(self, index):
        with open(self.dump_file, "r") as f:
            d = eval(f.read())
            print(d)
        return index in d

    def save_data(self):
        with open(self.dump_file, "w+") as f:
            f.write(str(self.data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', '--file', type=str, default="label.txt",
                        help='the directory to put in video')
    parser.add_argument('-n', '--names', type=list, default=["a", "b", "c", "d", "e", "f"],
                        help='the video name')
    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help='the video name')

    args = parser.parse_args()
    print(args.overwrite)
    labeler = Labeler(args.names, args.file, args.overwrite)
    labeler.get_data()
