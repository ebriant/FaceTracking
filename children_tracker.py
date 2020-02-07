import argparse
import json
import os
import config
import tensorflow as tf
import numpy as np
import utils as utils
from visualization import ImageProcessor
import logging
from PIL import Image
import matplotlib.image as mpimg
from data_handler import DataManager
from PyramidBox.preprocessing import ssd_vgg_preprocessing
from PyramidBox.nets.ssd import g_ssd_model
import PyramidBox.nets.np_methods as np_methods
from MemTrack.tracking.tracker import Tracker, Model

# TensorFlow session: grow memory when needed.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
conf = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=conf)

# Input placeholder.
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, data_format, resize=ssd_vgg_preprocessing.Resize.NONE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
predictions, localisations, _, end_points = g_ssd_model.get_model(image_4d)

# Restore SSD model.
ckpt_filename = 'PyramidBox/model/pyramidbox.ckpt'

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True

logging.basicConfig(level=config.logging_level)


class MainTracker:
    def __init__(self):

        self.viewer = None
        if args.out_img_dir is not None:
            self.out_img_dir = args.out_img_dir
        else:
            self.out_img_dir = os.path.join(args.out_dir, "img")

        self.model = None
        self.trackers_list = {}
        self.data_manager = DataManager(args.out_dir)
        # Load the video sequence
        self.s_frames = utils.get_video_frames(args.video, args.frames[0], args.frames[1])
        _, video_name = os.path.split(args.video)
        self.video_name = video_name[:-4]
        self.frame_data = {}
        self.last_frame_data = {}

        self.confidence = {}
        self.tmp_track = {}
        self.angular_order = []
        self.cur_img = None
        self.frame_number = args.frames[0]

    def write_frame_data(self, data=None):
        if data is None:
            data = self.frame_data
        file = "%s_%s.json" % (args.video, self.frame_number)
        path = os.path.join(args.out_dir, file)

        with open(path, 'w') as outfile:
            json.dump(data, outfile)
        print("file written")

    def start_tracking(self):
        # # Detect faces in the first image
        print(len(self.s_frames))
        self.cur_img = mpimg.imread(self.s_frames[0])
        self.cur_img = np.array(self.cur_img)
        self.viewer = ImageProcessor(self.cur_img, self.frame_number, cvt_color=True)

        if args.init is None:
            _, _, rbboxes = detect_faces(self.cur_img)
            bboxes_list = [utils.reformat_bbox_coord(bbox, 1) for bbox in rbboxes]

            # Let the user choose which faces to follow
            _, bboxes_list, names_list = self.viewer.select_bbox(bboxes_list)

            for idx, name in enumerate(names_list):
                self.last_frame_data[name] = {config.BBOX_KEY: bboxes_list[idx]}
        else:
            with open(args.init, 'r') as f:
                self.last_frame_data = json.load(f)

        self.data_manager.write_data(self.last_frame_data, self.video_name, self.frame_number)
        self.frame_number += 1

        angles_dict = utils.get_bbox_dict_ang_pos(self.last_frame_data)
        for name in sorted(angles_dict, key=angles_dict.get):
            self.angular_order.append(name)
        print(self.angular_order)
        self.angular_order = ['e', 'f', 'a', 'b', 'c', 'd']
        self.viewer.plt_frame_data(self.last_frame_data)
        self.viewer.show()
        self.viewer.save_img(self.out_img_dir)
        # Run the tracking process
        self.track_all()

    def set_up_tracker(self, name, bbox):
        tracker = Tracker(self.model)
        bbox = [int(i * self.viewer.size) for i in bbox]
        tracker.initialize(self.s_frames[0], bbox)
        self.trackers_list[name] = tracker

    def track_all(self):
        with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
            self.model = Model(sess)
            for name, data in self.last_frame_data.items():
                self.set_up_tracker(name, data[config.BBOX_KEY])

            while self.frame_number < args.frames[1]:
                logging.info("Processing frame {}".format(self.frame_number))

                for name, tracker in self.trackers_list.items():
                    tracker.idx += 1
                    bbox, cur_frame = tracker.track(self.s_frames[self.frame_number - args.frames[0]])
                    size = cur_frame.shape[0]
                    bbox = [i / size for i in bbox]
                    self.frame_data[name] = {config.BBOX_KEY: bbox}

                self.cur_img = cur_frame * 255
                self.viewer = ImageProcessor(self.cur_img, self.frame_number, cvt_color=True)
                # Check the overlay every frame
                issues = self.check_overlay()
                if issues:
                    self.correct_overlay(issues)

                if (self.frame_number - args.frames[0]) % args.rate == 0:
                    self.perform_corrections()

                self.viewer.plt_frame_data(self.frame_data)
                self.viewer.show()
                self.viewer.save_img(self.out_img_dir)
                self.data_manager.write_data(self.frame_data, self.video_name, self.frame_number)

                self.last_frame_data = self.frame_data.copy()
                self.frame_data = {}
                self.frame_number += 1
                # Check if the bbox is a face

            return

    def check_size(self):
        for name, data in self.frame_data.items():
            bbox = data[config.BBOX_KEY]
            if bbox[2] < config.min_bbox_size or bbox[3] < config.min_bbox_size:
                prev_bbox = self.last_frame_data[name][config.BBOX_KEY]
                self.correct_tracker(name, prev_bbox)

    def check_overlay(self):
        issues = []
        checked = []
        for name, data in self.frame_data.items():
            for name2, data2 in self.frame_data.items():
                if name != name2 and name2 not in checked and \
                        (utils.bb_intersection_over_union(data[config.BBOX_KEY], data2[
                            config.BBOX_KEY]) > config.tracking_overlay_threshold or utils.bb_contained(
                            data[config.BBOX_KEY],
                            data2[config.BBOX_KEY])):
                    logging.warning("Overlay issue between {}:{} and {}:{}".format(name, data[config.BBOX_KEY], name2,
                                                                                   data2[config.BBOX_KEY]))
                    issues.append((name, name2))
            checked.append(name)
        self.plot_overlay(issues)
        return issues

    def correct_overlay(self, issues):
        for issue in issues:
            name1 = issue[0]
            name2 = issue[1]
            data1 = self.frame_data[name1]
            data2 = self.frame_data[name2]

            bbox1 = data1[config.BBOX_KEY]
            bbox2 = data2[config.BBOX_KEY]
            prev_bbox1 = self.last_frame_data[name1][config.BBOX_KEY]
            prev_bbox2 = self.last_frame_data[name2][config.BBOX_KEY]

            iou1 = utils.bb_intersection_over_union(prev_bbox1, bbox1)
            iou2 = utils.bb_intersection_over_union(prev_bbox2, bbox2)

            if iou1 == iou2 == 0:
                self.correct_tracker(name1, prev_bbox1)
                self.correct_tracker(name2, prev_bbox2)
            elif iou1 > iou2:
                self.correct_tracker(name2, prev_bbox2)
            elif iou1 < iou2:
                self.correct_tracker(name1, prev_bbox1)
        return

    def plot_overlay(self, issues):
        color = (0, 255, 255)
        for issue in issues:
            bbox1 = self.frame_data[issue[0]][config.BBOX_KEY]
            bbox2 = self.frame_data[issue[1]][config.BBOX_KEY]
            vizu1 = [bbox1[0] - 0.005, bbox1[1] - 0.005, bbox1[2] + 0.01, bbox1[3] + 0.01]
            self.viewer.draw_bbox(vizu1, color=color, thickness=2)
            vizu2 = [bbox2[0] - 0.005, bbox2[1] - 0.005, bbox2[2] + 0.01, bbox2[3] + 0.01]
            self.viewer.draw_bbox(vizu2, color=color, thickness=2)
        return

    def perform_corrections(self):
        if not self.check_angular_order():
            logging.warning("angular order is broken")

        _, score_fd_list, bbox_fd_list = detect_faces(self.cur_img, select_threshold=config.face_detection_iou_trh)

        if len(bbox_fd_list) == 0:
            return

        bbox_fd_list = [utils.reformat_bbox_coord(bbox) for bbox in bbox_fd_list]

        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            if bbox_fd[2] < config.min_bbox_size or bbox_fd[3] < config.min_bbox_size:
                indices.append(idx)
        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        self.plot_fd_elements(bbox_fd_list, score_fd_list)
        bbox_fd_list, score_fd_list, corrected_bbox = self.iou_correction(bbox_fd_list, score_fd_list)
        bbox_fd_list, score_fd_list, corrected_bbox = self.roi_correction(bbox_fd_list, score_fd_list,
                                                                          corrected_bbox)
        bbox_fd_list, score_fd_list = self.cyclic_order_correction(bbox_fd_list, score_fd_list,
                                                                   corrected_bbox)
        if len(bbox_fd_list) > 0:
            logging.warning("Detected faces unused")

    def plot_fd_elements(self, bbox_fd_list, score_fd_list):
        # Draw detected faces
        for idx, bbox_fd in enumerate(bbox_fd_list):
            vizu = [bbox_fd[0] - 0.005, bbox_fd[1] - 0.005, bbox_fd[2] + 0.01, bbox_fd[3] + 0.01]
            self.viewer.draw_bbox(vizu, label=round(score_fd_list[idx], 3), color=(0, 125, 255), thickness=2)
        self.viewer.show()
        # Draw ROI
        for name, data in self.frame_data.items():
            bbox = data[config.BBOX_KEY]
            xmin, ymin, xmax, ymax = utils.get_roi(bbox, self.cur_img)
            roi = [xmin, ymin, xmax - xmin, ymax - ymin]
            self.viewer.draw_bbox(roi, color=(150, 0, 0))
        self.viewer.show()

    def iou_correction(self, bbox_fd_list, score_fd_list):
        corrected_bbox = {}
        if len(bbox_fd_list) == 0:
            return [], [], []
        candidates = {}
        match_count = {name: 0 for name in self.frame_data}
        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            candidates[idx] = []
            for name, data in self.frame_data.items():
                bbox = data[config.BBOX_KEY]
                iou = utils.bb_intersection_over_union(bbox, bbox_fd)
                if iou > config.correction_overlay_threshold:
                    candidates[idx].append((name, iou))
                    match_count[name] += 1

        for n, t in match_count.items():
            if t > 1:
                iou = 0
                for idx, c_list in candidates.items():
                    for i in c_list:
                        if i[0] == n and i[1] > iou:
                            iou = i[1]
                for idx, c_list in candidates.items():
                    candidates[idx] = [k for k in c_list if k[0] != n or k[1] == iou]

        for idx, c_list in candidates.items():
            if len(c_list) == 1:
                self.correct_tracker(c_list[0][0], bbox_fd_list[idx], face_confidence=True)
                corrected_bbox[c_list[0][0]] = {config.BBOX_KEY: bbox_fd_list[idx]}
                indices.append(idx)
            elif len(c_list) > 1:
                c = max(c_list, key=lambda x: x[1])
                self.correct_tracker(c[0], bbox_fd_list[idx], face_confidence=True)
                corrected_bbox[c[0]] = {config.BBOX_KEY: bbox_fd_list[idx]}
                indices.append(idx)
                logging.info("Bbox {} assigned to {} by max roi".format(bbox_fd_list[idx], c[0]))

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list, corrected_bbox

    def roi_correction(self, bbox_fd_list, score_fd_list, corrected_bbox):
        if len(bbox_fd_list) == 0:
            return [], [], []

        bbox_fd_list = [bbox_fd_list[idx] for idx, score in enumerate(score_fd_list) if
                        score > config.face_detection_roi_trh]
        score_fd_list = [score for score in score_fd_list if score > config.face_detection_roi_trh]

        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            for name, data in self.frame_data.items():
                bbox = data[config.BBOX_KEY]
                if name not in corrected_bbox and utils.bbox_in_roi(bbox, bbox_fd, self.cur_img) \
                        and self.check_angular_position(name, bbox_fd):
                    self.correct_tracker(name, bbox_fd, replace=True, face_confidence=True)
                    corrected_bbox[name] = {config.BBOX_KEY: bbox_fd}
                    indices.append(idx)
                    break

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list, corrected_bbox

    def cyclic_order_correction(self, bbox_fd_list, score_fd_list, corrected_bbox):
        if len(bbox_fd_list) == 0:
            return [], []

        bbox_fd_list = [bbox_fd_list[idx] for idx, score in enumerate(score_fd_list) if
                        score > config.face_detection_angle_trh]
        score_fd_list = [score for score in score_fd_list if score > config.face_detection_angle_trh]

        indices = []

        # Get the angles of the sure bboxes ordered in ---> corrected_bbox_angles
        corrected_bbox_angles_tmp = utils.get_bbox_dict_ang_pos(corrected_bbox)
        corrected_bbox_angles = {}
        verified_names = []
        for name in self.angular_order:
            if name in corrected_bbox_angles_tmp.keys():
                corrected_bbox_angles[name] = corrected_bbox_angles_tmp[name]
                verified_names.append(name)

        not_corrected_bbox_angles = {k: utils.get_angle(v[config.BBOX_KEY]) for k, v in
                                     self.frame_data.items() if k not in corrected_bbox}
        if not not_corrected_bbox_angles:
            return [], []

        bbox_fd_angles = [utils.get_angle(bbox_fd) for bbox_fd in bbox_fd_list]

        tmp_order = {}
        l = len(verified_names)
        for idx, angle in enumerate(bbox_fd_angles):
            for i in range(l):
                if utils.is_between(corrected_bbox_angles[verified_names[i]],
                                    corrected_bbox_angles[verified_names[(i + 1) % l]],
                                    angle):
                    tmp_order[idx] = (verified_names[i], verified_names[(i + 1) % l])
                    break

        for idx, bbox_fd in enumerate(bbox_fd_list):
            angle = utils.get_angle(bbox_fd)
            if corrected_bbox_angles:
                prev_id, next_id = None, None
                for name, value in corrected_bbox_angles.items():
                    if angle > value:
                        prev_id = name
                        break
                for name, value in corrected_bbox_angles.items():
                    if angle < value:
                        next_id = name
                        break
                if prev_id is None:
                    prev_id = list(corrected_bbox_angles.keys())[-1]
                if next_id is None:
                    next_id = list(corrected_bbox_angles.keys())[0]
                ang_order = self.angular_order * 2
                start = ang_order.index(prev_id)
                end = ang_order.index(next_id, start + 1)
                potential_id_list = [i for i in ang_order[start + 1:end]]
            else:
                potential_id_list = self.angular_order

            if self.check_angle_proximity(angle, corrected_bbox_angles):
                name = None
                if len(potential_id_list) == 1 and self.check_angular_position(potential_id_list[0], bbox_fd):
                    name = potential_id_list[0]
                elif len(potential_id_list) == 0:
                    continue
                else:
                    key, value = min(not_corrected_bbox_angles.items(), key=lambda kv: abs(kv[1] - angle))
                    if self.check_angular_position(key, bbox_fd):
                        name = key

                if name is not None:
                    self.correct_tracker(name, bbox_fd, replace=True, face_confidence=True)
                    indices.append(idx)
                    logging.info("Assigned {} to children {} by closest angular position".format(bbox_fd, name))

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list

    def get_id_between(self, id1, id2):
        idx1, idx2 = self.angular_order.index(id1) + 1, self.angular_order.index(id2)
        if idx1 <= idx2:
            return self.angular_order[idx1:idx2]
        else:
            return self.angular_order[idx1:] + self.angular_order[:idx2]

    def get_bbox_between_id(self, id1, id2, bbox_fd_list):
        a1 = utils.get_angle(self.frame_data[id1][config.BBOX_KEY])
        a2 = utils.get_angle(self.frame_data[id2][config.BBOX_KEY])
        bbox_list = [i for i in bbox_fd_list if utils.is_between(a1, a2, utils.get_angle(i))]
        angle_list = []

        for bbox_fd in bbox_fd_list:
            angle = utils.get_angle(bbox_fd)
            if utils.is_between(a1, a2, angle):
                bbox_list.append(bbox_fd)
                angle_list.append(angle)
                zipped_pairs = zip(angle_list, bbox_list)
                zipped_pairs = sorted(zipped_pairs)
                bbox_list = [x for _, x in zipped_pairs]
                angle_list = [x for x, _ in zipped_pairs]

        return bbox_list, angle_list

    def check_angle_proximity(self, angle, angles_dict):
        for name, angle2 in angles_dict.items():
            a = abs((angle - angle2 + 180) % 360 - 180)
            if a < config.angle_proximity_treshhold:
                return False
        return True

    def correct_tracker(self, name, bbox, *, replace=False, face_confidence=False):
        self.frame_data[name][config.BBOX_KEY] = bbox
        bbox_px = [int(i * self.viewer.size) for i in bbox]
        if face_confidence:
            self.frame_data[name][config.FACE_CONFIDENCE_KEY] = 1

        if replace:
            tracker = Tracker(self.model)
            tracker.initialize(self.s_frames[self.frame_number - args.frames[0]], bbox_px)
            self.trackers_list[name] = tracker
        else:
            self.trackers_list[name].redefine_roi(bbox_px)

    def check_face(self, bbox, fd_bbox_list, name):
        corrected_bbox = []
        for bbox_fd in fd_bbox_list:
            if utils.bb_intersection_over_union(bbox, bbox_fd) > config.correction_overlay_threshold:
                return False, bbox_fd
            elif ((name2 != name and utils.bb_intersection_over_union(self.frame_data[name2][config.BBOX_KEY],
                                                                      bbox_fd) < config.correction_overlay_threshold)
                  for name2 in
                  self.frame_data):
                corrected_bbox.append(bbox_fd)

        if len(corrected_bbox) == 0:
            return True, None
        elif len(corrected_bbox) > 1:
            # TODO: Do something if several corrections possible
            logging.warning("Several correction possible")
            return False, corrected_bbox[0]
        else:
            return False, corrected_bbox[0]

    def check_angular_position(self, name, bbox):
        angle = utils.get_angle(bbox)
        l = len(self.angular_order)
        ang_order = self.angular_order * 2
        idx = ang_order.index(name)
        prev = self.angular_order[(idx - 1) % l]
        next = self.angular_order[(idx + 1) % l]
        angles_dict = utils.get_bbox_dict_ang_pos(self.frame_data)
        start, end = angles_dict[prev], angles_dict[next]
        return utils.is_between(start, end, angle)

    def check_angular_order(self):
        angles_dict = utils.get_bbox_dict_ang_pos(self.frame_data)
        tmp_order = []
        for name in sorted(angles_dict, key=angles_dict.get):
            tmp_order.append(name)
        start = self.angular_order.index(tmp_order[0])
        l = len(self.angular_order)
        for i, n in enumerate(tmp_order):
            if n != self.angular_order[(start + i) % l]:
                return False
        return True


def detect_faces(img, select_threshold=0.35, nms_threshold=0.1):
    # Run SSD network.
    h, w = img.shape[:2]
    if h < w and h < 640:
        scale = 640. / h
        h = 640
        w = int(w * scale)
    elif h >= w and w < 640:
        scale = 640. / w
        w = 640
        h = int(h * scale)
    img = Image.fromarray(np.uint8(img))
    resized_img = img.resize((w, h))
    net_shape = np.array(resized_img).shape[:2]
    rimg, rpredictions, rlocalisations, rbbox_img, e_ps = isess.run(
        [image_4d, predictions, localisations, bbox_img, end_points], feed_dict={img_input: resized_img})

    layer_shape = [e_ps['block3'].shape[1:3], e_ps['block4'].shape[1:3], e_ps['block5'].shape[1:3],
                   e_ps['block7'].shape[1:3], e_ps['block8'].shape[1:3], e_ps['block9'].shape[1:3]]

    # SSD default anchor boxes.
    ssd_anchors = g_ssd_model.ssd_anchors_all_layers(feat_shapes=layer_shape, img_shape=net_shape)

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations[0], ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=1200)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    return rclasses, rscores, rbboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')
    parser.add_argument('-v', '--video', type=str, required=True, help="video to use")
    parser.add_argument('-f', '--frames', nargs='+', type=int, required=True, help="frames range")
    parser.add_argument('-o', '--out_dir', type=str, required=True, help="Output directory")
    parser.add_argument('-i', '--init', type=str, help="initialization file")
    parser.add_argument('--out_img_dir', type=str, help="Output directory for img")
    parser.add_argument('-r', '--rate', type=int, default=30, help='rate of correction application')

    args = parser.parse_args()
    main_tracker = MainTracker()
    main_tracker.start_tracking()
