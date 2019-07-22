import argparse
import config
import utils
import pprint
import matplotlib.image as mpimg
import numpy as np
import visualization

FRAME_COUNT = 9999
DECIMAL_PRECISION = 4
THRESHOLD = 0.5
GT_COLOR = (0,255,0)
DATA_COLOR = (0,0,255)



class Evaluator:
    def __init__(self, labels_file, data_file):
        with open(labels_file) as f:
            self.labels = eval(f.read())
        with open(data_file) as f:
            self.data = eval(f.read())
        self.perf = {}

    def get_performances_bbox(self):
        right_count = {}
        label_count = {}
        for name in next(iter(self.labels.values())):
            right_count[name] = 0
            label_count[name] = 0
        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        general_count = 0

        print(data_length, len(self.labels))

        nb_frame = 0
        nb_children = len(right_count)
        for frame, frame_label in self.labels.items():
            if frame > data_length or frame > FRAME_COUNT:
                break
            nb_frame += 1
            for name, label in frame_label.items():
                if (label[2] > 0 and label[3] > 0):
                    label_count[name] += 1
                    right_count[name] += \
                        1 if utils.bb_intersection_over_union(self.data[name][config.BBOX_KEY][frame], label) > THRESHOLD \
                        else 0
                    general_count += 1 if utils.is_bbox_in_bbox_list(
                            [self.data[name][config.BBOX_KEY][frame] for name in self.data], label, THRESHOLD) \
                            else 0

        label_count_total = 0
        for key, value in label_count.items():
            label_count_total += value
        self.perf = {
            "frame_number": nb_frame,
            "children_number": nb_children,
            "general_accuracy": round(general_count / label_count_total, DECIMAL_PRECISION),
            "label_percentage": round(label_count_total/(nb_frame * nb_children), DECIMAL_PRECISION)
        }

        sum = 0
        for name in right_count:
            name_key = "child_" + name
            self.perf[name_key] = round(right_count[name] / label_count[name], DECIMAL_PRECISION)
            sum += self.perf[name_key]

        self.perf["average_accuracy"] = round(sum / nb_children, DECIMAL_PRECISION)

    def get_performances(self):
        right_count = {}
        for name in next(iter(self.labels.values())):
            right_count[name] = 0

        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        general_count = 0

        print(data_length, len(self.labels))

        nb_frame = 0
        nb_children = len(right_count)
        for frame, frame_label in self.labels.items():
            if frame > data_length or frame > FRAME_COUNT:
                break
            nb_frame += 1
            for name, label in frame_label.items():
                if utils.is_point_in_bbox(self.data[name][config.BBOX_KEY][frame], label):
                    right_count[name] += 1

                if utils.is_point_in_bbox_list([self.data[name][config.BBOX_KEY][frame] for name in self.data], label):
                    general_count += 1

        self.perf = {
            "frame_number": nb_frame,
            "children_number": nb_children,
            "general_accuracy": round(general_count / (nb_frame * nb_children), DECIMAL_PRECISION)
        }

        sum = 0
        for name in right_count:
            name_key = "child_" + name
            self.perf[name_key] = round(right_count[name] / nb_frame, DECIMAL_PRECISION)
            sum += self.perf[name_key]

        self.perf["average_accuracy"] = round(sum / nb_children, DECIMAL_PRECISION)

    def make_video(self, out_dir):
        visualizer = visualization.VisualizerOpencv()
        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        s_frames = utils.load_seq_video()
        for frame, frame_label in self.labels.items():
            if frame > data_length or frame > FRAME_COUNT:
                break
            img = mpimg.imread(s_frames[frame])
            img = np.array(img)
            visualizer.prepare_img(img, frame)
            for name, label in frame_label.items():
                visualizer.draw_bbox(label, name, color=GT_COLOR)
                visualizer.draw_bbox(self.data[name][config.BBOX_KEY][frame], name, color=DATA_COLOR)
            visualizer.save_img(out_dir)


e = Evaluator("data/labels/171214_1_test.txt", "data/output/171214_1.txt")
e.get_performances_bbox()
pprint.pprint(e.perf)

e = Evaluator("data/labels/171214_1_test.txt", "data/output/171214_1_tracking.txt")
e.get_performances_bbox()
pprint.pprint(e.perf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-l', '--labels', type=str, help="path to labels file")
    parser.add_argument('-d', '--data', type=str, help="path to data file")
    parser.add_argument('-o', '--output', type=str, help="output path")
    parser.add_argument('-m', '--mode', type=str, choices=['eval', 'eval_bbox', 'visu'], help="mode of operation")

    args = parser.parse_args()
    e = Evaluator(args.labels, args.data)
    if args.mode == "eval":
        e.get_performances()
        pprint.pprint(e.perf)
    elif args.mode == "eval_bbox":
        e.get_performances_bbox()
        pprint.pprint(e.perf)
    elif args.mode == "visu":
        e.make_video(args.output)


# e = Evaluator("data/labels/171214_1.txt", "data/output/171214_1_overlay.txt")
# e.get_performances()
# pprint.pprint(e.perf)
#
# e = Evaluator("data/labels/171214_2.txt", "data/output/171214_2.txt")
# e.get_performances()
# pprint.pprint(e.perf)
#
# e = Evaluator("data/labels/171214_2.txt", "data/output/171214_2_tracking.txt")
# e.get_performances()
# pprint.pprint(e.perf)
