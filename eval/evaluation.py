import os
import config
import utils


class Evaluator:
    def __init__(self, labels_file, data_file):
        with open(labels_file) as f:
            self.labels = eval(f.read())
        with open(data_file) as f:
            self.data = eval(f.read())
        self.perf = {}
        print({name:{config.BBOX_KEY:self.data[name][config.BBOX_KEY][3515]} for name in self.data})

    def get_performances(self):
        right_count = {}
        for name in next(iter(self.labels.values())):
            right_count[name] = 0

        data_length = len(next(iter(self.data.values()))[config.BBOX_KEY])
        general_count = 0

        nb_frame = 0
        nb_children = len(right_count)
        for frame, frame_label in self.labels.items():
            if frame > data_length:
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
            "general_accuracy": general_count / (nb_frame*nb_children)
        }
        sum = 0
        for name in right_count:
            self.perf[name] = right_count[name] / nb_frame
            sum += self.perf[name]

        self.perf["average_accuracy"] = sum/nb_children

e = Evaluator("data/labels/171214_1.txt", "data/output/171214_1_verif30/171214_1.txt")
e.get_performances()
print(e.perf)
