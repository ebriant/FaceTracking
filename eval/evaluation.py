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
GT_COLOR = (0, 255, 0)
DATA_COLOR = (0, 0, 255)


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
        print(data_length)
        print(len(self.labels))
        general_count = 0

        nb_frame = 0
        nb_children = len(right_count)
        # for frame in range(max(data_length, FRAME_COUNT)):
        #     frame_label = self.labels[frame]

        for frame, frame_label in self.labels.items():
            if frame > data_length or frame > FRAME_COUNT:
                continue
            nb_frame += 1
            for name, label in frame_label.items():
                if label[2] > 0 and label[3] > 0:
                    label_count[name] += 1
                    right_count[name] += \
                        1 if utils.bb_intersection_over_union(self.data[name][config.BBOX_KEY][frame], label) \
                             > THRESHOLD else 0

                    general_count += 1 if utils.is_bbox_in_bbox_list(
                        [self.data[name][config.BBOX_KEY][frame] for name in self.data], label, THRESHOLD)[0] \
                        else 0

        label_count_total = 0
        for key, value in label_count.items():
            label_count_total += value
        self.perf = {
            "frame_number": nb_frame,
            "children_number": nb_children,
            "general_accuracy": round(general_count / label_count_total, DECIMAL_PRECISION),
            "label_percentage": round(label_count_total / (nb_frame * nb_children), DECIMAL_PRECISION)
        }

        sum = 0
        for name in right_count:
            name_key = "child_" + name
            self.perf[name_key] = round(right_count[name] / label_count[name], DECIMAL_PRECISION)
            sum += self.perf[name_key]

        self.perf["average_accuracy"] = round(sum / nb_children, DECIMAL_PRECISION)

    def get_performances_bbox_position_only(self):
        nb_children = len(next(iter(self.labels.values())))
        label_count = {}
        for name in next(iter(self.labels.values())):
            label_count[name] = 0
        data_length = len(self.data)
        general_count = 0
        nb_frame = 0

        for frame_idx, frame_data in self.data.items():
            if frame_idx > FRAME_COUNT or frame_idx not in self.labels:
                continue
            nb_frame += 1
            labels = [self.labels[frame_idx][name] for name in self.labels[frame_idx]]
            for name, label in self.labels[frame_idx].items():
                if label[2] > 0 and label[3] > 0:
                    label_count[name] += 1
            for bbox in frame_data:
                result, match = utils.is_bbox_in_bbox_list(labels, bbox, THRESHOLD)
                if result:
                    general_count += 1
                    labels.remove(match)

        label_count_total = 0
        for key, value in label_count.items():
            label_count_total += value
        self.perf = {
            "frame_number": nb_frame,
            "general_accuracy": round(general_count / label_count_total, DECIMAL_PRECISION)
        }

    def get_performances(self):
        right_count = {}
        for name in next(iter(self.labels.values())):
            right_count[name] = 0

        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        general_count = 0

        nb_frame = 0
        nb_children = len(right_count)
        for frame, frame_label in self.labels.items():
            if frame > data_length or frame > FRAME_COUNT:
                continue
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
        visualizer = visualization.ImageProcessor()
        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        s_frames = utils.get_video_frames()
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


print("Video1")

e = Evaluator("data/labels/171214_1_bbox.txt", "data/output/171214_1.txt")
e.get_performances_bbox()
print(e.perf["average_accuracy"])
pprint.pprint(e.perf)
print("####")

e = Evaluator("data/labels/171214_1_bbox.txt", "data/output/171214_1_tracking.txt")
e.get_performances_bbox()
print(e.perf["average_accuracy"])
pprint.pprint(e.perf)
print("####")

e = Evaluator("data/labels/171214_1_bbox.txt", "data/output/171214_1_fd.txt")
e.get_performances_bbox_position_only()
# print(e.perf["average_accuracy"])
pprint.pprint(e.perf)

print("Video2")

e = Evaluator("data/labels/171214_2_bbox.txt", "data/output/171214_2.txt")
e.get_performances_bbox()
print(e.perf["average_accuracy"])
pprint.pprint(e.perf)
print("####")

e = Evaluator("data/labels/171214_2_bbox.txt", "data/output/171214_2_tracking.txt")
e.get_performances_bbox()
print(e.perf["average_accuracy"])
pprint.pprint(e.perf)
print("####")

e = Evaluator("data/labels/171214_2_bbox.txt", "data/output/171214_2_fd.txt")
e.get_performances_bbox_position_only()
# print(e.perf["average_accuracy"])
pprint.pprint(e.perf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-l', '--labels', type=str, help="path to labels file")
    parser.add_argument('-d', '--data', type=str, help="path to data file")
    parser.add_argument('-o', '--output', type=str, help="output path")
    parser.add_argument('-m', '--mode', type=str, choices=['eval', 'eval_bbox', 'visu'], help="mode of operation")

    # args = parser.parse_args()
    # e = Evaluator(args.labels, args.data)
    # if args.mode == "eval":
    #     e.get_performances()
    #     pprint.pprint(e.perf)
    # elif args.mode == "eval_bbox":
    #     e.get_performances_bbox()
    #     pprint.pprint(e.perf)
    # elif args.mode == "visu":
    #     e.make_video(args.output)
