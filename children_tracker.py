import os
import config
import tensorflow as tf
import numpy as np
import utils
import logging
from PIL import Image
import matplotlib.image as mpimg
import visualization
import face_alignment

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

START_FRAME = 1


class MainTracker:
    def __init__(self):
        self.model = None
        self.visualizer = visualization.VisualizerOpencv()
        self.face_aligner = face_alignment.FaceAligner()
        self.trackers_list = {}
        self.lt_trackers_list = {}

        # Load the video sequence
        self.s_frames = utils.get_video_frames()
        if not os.path.exists(config.out_dir):
            os.mkdir(config.out_dir)
        _, video_name = os.path.split(config.video_path)
        self.dump_file_path = os.path.join(config.out_dir, "{}.txt".format(video_name[:-4]))

        self.confidence = {}
        self.data = {}
        self.tmp_track = {}
        self.latent_track = {}
        self.angular_order = []
        self.cur_img = None
        self.frame_number = 0

    def save_data(self):
        with open(self.dump_file_path, "w+") as f:
            f.write(str(self.data))

    def start_tracking(self):
        # # Detect faces in the first image
        self.cur_img = mpimg.imread(self.s_frames[0])
        self.cur_img = np.array(self.cur_img)

        if config.init is None:
            _, _, rbboxes = detect_faces(self.cur_img)
            bboxes_list = [utils.reformat_bbox_coord(bbox, self.cur_img.shape[0]) for bbox in rbboxes]

            # Let the user choose which faces to follow
            self.visualizer.prepare_img(self.cur_img, 0)
            _, bboxes_list, names_list = self.visualizer.select_bbox(bboxes_list)
            for idx, name in enumerate(names_list):
                self.tmp_track[name] = {config.BBOX_KEY: bboxes_list[idx]}
                self.data[name] = {config.BBOX_KEY: []}
            self.merge_temp()
            print(self.tmp_track)

        else:
            self.tmp_track = config.init
            for name in self.tmp_track:
                self.data[name] = {config.BBOX_KEY: []}
            self.merge_temp()

        for name in self.tmp_track:
            self.confidence[name] = 1
        angles_dict = utils.get_bbox_dict_ang_pos(self.tmp_track, self.cur_img.shape)
        for name in sorted(angles_dict, key=angles_dict.get):
            self.angular_order.append(name)

        # Run the tracking process
        self.track_all()

    def set_up_tracker(self, name, bbox):
        tracker = Tracker(self.model)
        tracker.initialize(self.s_frames[self.frame_number], bbox)
        self.trackers_list[name] = tracker
        self.lt_trackers_list[name] = tracker

    def track_all(self):
        with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
            self.model = Model(sess)
            for name, data in self.data.items():
                self.set_up_tracker(name, data[config.BBOX_KEY][0])

            frame_idx = START_FRAME
            while frame_idx < len(self.s_frames):
                self.tmp_track = {}
                last_frame = min(frame_idx + config.checking_rate, len(self.s_frames))

                for idx in range(frame_idx, last_frame):
                    self.frame_number = idx
                    logging.info("Processing frame {}".format(idx))
                    for name, tracker in self.trackers_list.items():
                        tracker.idx += 1
                        bbox, cur_frame = tracker.track(self.s_frames[self.frame_number])
                        bbox = [int(i) for i in bbox]
                        self.tmp_track[name] = {config.BBOX_KEY: bbox}

                    self.cur_img = cur_frame * 255
                    self.visualizer.prepare_img(self.cur_img, idx)

                    # Check the overlay every frame
                    issues = self.check_overlay()
                    if issues:
                        self.correct_overlay(issues)
                    if idx != last_frame - 1:
                        self.visualizer.plt_img(self.tmp_track)
                        self.visualizer.save_img(config.out_dir)

                        self.merge_temp()

                # Check if the bbox is a face
                frame_idx = last_frame
                self.check_faces()
                self.merge_temp()
                # Visualization
                self.visualizer.plt_img(self.tmp_track)
                self.visualizer.save_img(config.out_dir)
                self.save_data()
            return

    def update_confidence(self, name, confidence=0.0):
        self.confidence[name] = round(self.confidence[name] * (1 - config.conf_ud_rt) + confidence * config.conf_ud_rt,4)

    def merge_temp(self):
        for name, data in self.tmp_track.items():
            self.data[name][config.BBOX_KEY].append(list(self.tmp_track[name][config.BBOX_KEY]))

    def check_size(self):
        for name, data in self.tmp_track.items():
            bbox = data[config.BBOX_KEY]
            if bbox[2] < config.min_bbox_size or bbox[3] < config.min_bbox_size:
                prev_bbox = self.data[name][config.BBOX_KEY][-1]
                self.correct_tracker(name, prev_bbox)

    def check_overlay(self):
        issues = []
        checked = []
        for name, data in self.tmp_track.items():
            for name2, data2 in self.tmp_track.items():
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
            data1 = self.tmp_track[name1]
            data2 = self.tmp_track[name2]

            bbox1 = data1[config.BBOX_KEY]
            bbox2 = data2[config.BBOX_KEY]
            prev_bbox1 = self.data[name1][config.BBOX_KEY][-1]
            prev_bbox2 = self.data[name2][config.BBOX_KEY][-1]

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
            bbox1 = self.tmp_track[issue[0]][config.BBOX_KEY]
            bbox2 = self.tmp_track[issue[1]][config.BBOX_KEY]
            vizu1 = [bbox1[0] - 2, bbox1[1] - 2, bbox1[2] + 4, bbox1[3] + 4]
            self.visualizer.draw_bbox(vizu1, color=color, thickness=2)
            vizu2 = [bbox2[0] - 2, bbox2[1] - 2, bbox2[2] + 4, bbox2[3] + 4]
            self.visualizer.draw_bbox(vizu2, color=color, thickness=2)
        return

    def check_faces(self):
        if not self.check_angular_order():
            logging.warning("angular order is broken")
        _, score_fd_list, bbox_fd_list = detect_faces(self.cur_img, select_threshold=config.face_detection_iou_trh)

        if len(bbox_fd_list) == 0:
            return

        bbox_fd_list = [utils.reformat_bbox_coord(bbox, self.cur_img.shape[0]) for bbox in bbox_fd_list]

        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            if bbox_fd[2] < config.min_bbox_size or bbox_fd[3] < config.min_bbox_size:
                indices.append(idx)
        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]

        self.plot_fd_elements(bbox_fd_list, score_fd_list)

        # self.long_term_tracker()
        bbox_fd_list, score_fd_list, corrected_bbox = self.correct_faces_by_iou(bbox_fd_list, score_fd_list)
        bbox_fd_list, score_fd_list, corrected_bbox = self.correct_faces_by_roi(bbox_fd_list, score_fd_list,
                                                                                corrected_bbox)
        bbox_fd_list, score_fd_list = self.correct_faces_by_proximity(bbox_fd_list, score_fd_list,
                                                                              corrected_bbox)

        for name, data in self.tmp_track.items():
            if name not in corrected_bbox:
                self.update_confidence(name, 0)

        logging.info(self.confidence)
        if len(bbox_fd_list) > 0:
            logging.warning("Detected faces unused")

        self.latent_track = self.tmp_track.copy()

    def long_term_tracker(self):
        for name, tracker in self.lt_trackers_list.items():
            tracker.idx += 1
            bbox, cur_frame = tracker.track(self.s_frames[self.frame_number])
            bbox = [int(i) for i in bbox]
            self.visualizer.draw_bbox(bbox, color=(255, 255, 0), thickness=2)
        self.visualizer.plt_img({})

    def plot_fd_elements(self, bbox_fd_list, score_fd_list):
        # Draw detected faces
        for idx, bbox_fd in enumerate(bbox_fd_list):
            vizu = [bbox_fd[0] - 2, bbox_fd[1] - 2, bbox_fd[2] + 4, bbox_fd[3] + 4]
            self.visualizer.draw_bbox(vizu, label=round(score_fd_list[idx], 3), color=(0, 125, 255), thickness=2)
        self.visualizer.plt_img({})

        # Draw ROI
        for name, data in self.tmp_track.items():
            bbox = data[config.BBOX_KEY]
            xmin, ymin, xmax, ymax = utils.get_roi(bbox, self.cur_img)
            roi = [xmin, ymin, xmax - xmin, ymax - ymin]
            self.visualizer.draw_bbox(roi, color=(150, 0, 0))
        self.visualizer.plt_img({})

    def correct_faces_by_iou(self, bbox_fd_list, score_fd_list):
        corrected_bbox = {}
        if len(bbox_fd_list) == 0:
            return [], [], []
        candidates = {}
        match_count = {name: 0 for name in self.tmp_track}
        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            candidates[idx] = []
            for name, data in self.tmp_track.items():
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
                self.update_confidence(c_list[0][0], 1)
                self.correct_tracker(c_list[0][0], bbox_fd_list[idx])
                corrected_bbox[c_list[0][0]] = {config.BBOX_KEY: bbox_fd_list[idx]}
                indices.append(idx)
            elif len(c_list) > 1:
                c = max(c_list, key=lambda x: x[1])
                self.correct_tracker(c[0], bbox_fd_list[idx])
                corrected_bbox[c[0]] = {config.BBOX_KEY: bbox_fd_list[idx]}
                self.update_confidence(c[0], 1)
                indices.append(idx)
                logging.info("Bbox {} assigned to {} by max roi".format(bbox_fd_list[idx], c[0]))

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list, corrected_bbox

    def correct_faces_by_roi(self, bbox_fd_list, score_fd_list, corrected_bbox):
        if len(bbox_fd_list) == 0:
            return [], [], []

        bbox_fd_list = [bbox_fd_list[idx] for idx, score in enumerate(score_fd_list) if
                        score > config.face_detection_roi_trh]
        score_fd_list = [score for score in score_fd_list if score > config.face_detection_roi_trh]

        indices = []
        for idx, bbox_fd in enumerate(bbox_fd_list):
            for name, data in self.tmp_track.items():
                bbox = data[config.BBOX_KEY]
                if name not in corrected_bbox and utils.bbox_in_roi(bbox, bbox_fd, self.cur_img) \
                        and self.check_angular_position(name, bbox_fd):
                    self.correct_tracker(name, bbox_fd, True)
                    self.update_confidence(name, 0.5)
                    corrected_bbox[name] = {config.BBOX_KEY: bbox_fd}
                    indices.append(idx)
                    break

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list, corrected_bbox

    def correct_faces_by_proximity(self, bbox_fd_list, score_fd_list, corrected_bbox):
        if len(bbox_fd_list) == 0:
            return [], []

        bbox_fd_list = [bbox_fd_list[idx] for idx, score in enumerate(score_fd_list) if
                        score > config.face_detection_angle_trh]
        score_fd_list = [score for score in score_fd_list if score > config.face_detection_angle_trh]

        indices = []

        # Get the angles of the sure bboxes ordered in ---> corrected_bbox_angles
        corrected_bbox_angles_tmp = utils.get_bbox_dict_ang_pos(corrected_bbox, self.cur_img.shape)
        corrected_bbox_angles = {}
        verified_names = []
        for name in self.angular_order:
            if name in corrected_bbox_angles_tmp.keys():
                corrected_bbox_angles[name] = corrected_bbox_angles_tmp[name]
                verified_names.append(name)

        not_corrected_bbox_angles = {k: utils.get_angle(v[config.BBOX_KEY], self.cur_img.shape) for k, v in
                                     self.tmp_track.items() if k not in corrected_bbox}
        if not not_corrected_bbox_angles:
            return [], []

        bbox_fd_angles = [utils.get_angle(bbox_fd, self.cur_img.shape) for bbox_fd in bbox_fd_list]

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
            angle = utils.get_angle(bbox_fd, self.cur_img.shape)
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
                    self.correct_tracker(name, bbox_fd, True)
                    indices.append(idx)
                    logging.info("Assigned {} to children {} by closest angular position".format(bbox_fd, name))

        bbox_fd_list = [i for j, i in enumerate(bbox_fd_list) if j not in indices]
        score_fd_list = [i for j, i in enumerate(score_fd_list) if j not in indices]
        return bbox_fd_list, score_fd_list

    def correct_faces_by_order_association(self, bbox_fd_list, score_fd_list, corrected_bbox):
        bbox_fd_list = [bbox_fd_list[idx] for idx, score in enumerate(score_fd_list) if
                        score > config.face_detection_angle_trh]
        score_fd_list = [score for score in score_fd_list if score > config.face_detection_angle_trh]

        corrected_bbox = [i for i in self.angular_order if i in corrected_bbox]
        l = len(corrected_bbox)
        for i in range(l):
            if len(bbox_fd_list) == 0:
                break
            id1, id2 = corrected_bbox[i], corrected_bbox[(i + 1) % l]
            candidates_bbox, angles = self.get_bbox_between_id(id1, id2, bbox_fd_list)
            candidates_id = self.get_id_between(id1, id2)
            if len(candidates_bbox) == len(candidates_id):
                for index, id in enumerate(candidates_id):
                    self.correct_tracker(id, candidates_bbox[index])

            bbox_fd_list = [i for i in bbox_fd_list if i not in candidates_bbox]
        return bbox_fd_list, score_fd_list

    def recover_id(self):
        return

    def get_id_between(self, id1, id2):
        idx1, idx2 = self.angular_order.index(id1) + 1, self.angular_order.index(id2)
        if idx1 <= idx2:
            return self.angular_order[idx1:idx2]
        else:
            return self.angular_order[idx1:] + self.angular_order[:idx2]

    def get_bbox_between_id(self, id1, id2, bbox_fd_list):
        a1 = utils.get_angle(self.tmp_track[id1][config.BBOX_KEY], self.cur_img.shape)
        a2 = utils.get_angle(self.tmp_track[id2][config.BBOX_KEY], self.cur_img.shape)
        bbox_list = [i for i in bbox_fd_list if utils.is_between(a1, a2, utils.get_angle(i, self.cur_img.shape))]
        angle_list = []

        for bbox_fd in bbox_fd_list:
            angle = utils.get_angle(bbox_fd, self.cur_img.shape)
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

    def correct_tracker(self, name, bbox, replace=False):
        self.tmp_track[name][config.BBOX_KEY] = bbox
        if replace:
            tracker = Tracker(self.model)
            tracker.initialize(self.s_frames[self.frame_number], bbox)
            self.trackers_list[name] = tracker
        else:
            self.trackers_list[name].redefine_roi(bbox)

    def check_face(self, bbox, fd_bbox_list, name):
        corrected_bbox = []
        for bbox_fd in fd_bbox_list:
            if utils.bb_intersection_over_union(bbox, bbox_fd) > config.correction_overlay_threshold:
                return False, bbox_fd
            elif ((name2 != name and utils.bb_intersection_over_union(self.tmp_track[name2][config.BBOX_KEY],
                                                                      bbox_fd) < config.correction_overlay_threshold)
                  for name2 in
                  self.tmp_track):
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
        angle = utils.get_angle(bbox, self.cur_img.shape)
        l = len(self.angular_order)
        ang_order = self.angular_order * 2
        idx = ang_order.index(name)
        prev = self.angular_order[(idx - 1) % l]
        next = self.angular_order[(idx + 1) % l]
        angles_dict = utils.get_bbox_dict_ang_pos(self.tmp_track, self.cur_img.shape)
        start, end = angles_dict[prev], angles_dict[next]
        return utils.is_between(start, end, angle)

    def check_angular_order(self):
        angles_dict = utils.get_bbox_dict_ang_pos(self.tmp_track, self.cur_img.shape)
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
    main_tracker = MainTracker()
    main_tracker.start_tracking()
