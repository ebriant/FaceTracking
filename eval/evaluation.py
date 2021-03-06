import config
import utils
import pprint

FRAME_COUNT = 9999
DECIMAL_PRECISION = 4
THRESHOLD = 0.4

class Evaluator:
    def __init__(self, labels_file, data_file):
        with open(labels_file) as f:
            self.labels = eval(f.read())
        with open(data_file) as f:
            self.data = eval(f.read())
        self.perf = {}
        # print({name:{config.BBOX_KEY:self.data[name][config.BBOX_KEY][4980]} for name in self.data})

    def get_performances_bbox(self):
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
                if utils.bb_intersection_over_union(self.data[name][config.BBOX_KEY][frame], label) > THRESHOLD:
                    right_count[name] += 1
                if utils.is_bbox_in_bbox_list([self.data[name][config.BBOX_KEY][frame] for name in self.data], label, THRESHOLD):
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


e = Evaluator("data/labels/171214_1_test.txt", "data/output/171214_1.txt")
e.get_performances_bbox()
pprint.pprint(e.perf)

# e = Evaluator("data/labels/171214_1.txt", "data/output/171214_1_tracking.txt")
# e.get_performances()
# pprint.pprint(e.perf)
#
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
