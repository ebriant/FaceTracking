import os
import cv2
import config
import tensorflow as tf
import numpy as np
import utils

from PIL import Image
import matplotlib.image as mpimg
import visualization
import data_handler
import face_alignment
from faceAlignment.face_alignment.api import FaceAlignment, LandmarksType

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


class MainTracker:
    def __init__(self):
        self.data_handler = data_handler.DataHandler("_")
        self.visualizer = visualization.VisualizerOpencv()
        self.face_aligner = face_alignment.FaceAligner()
        self.trackers_list = []
        self.fa = FaceAlignment(LandmarksType._3D, device='cuda:0', flip_input=True)
        # Load the video sequence
        self.s_frames = load_seq_video()
        self.landmarks = {}
        self.data = {}

    def start_tracking(self):
        # Detect faces in the first image
        img = mpimg.imread(self.s_frames[0])
        img = np.array(img)
        _, _, rbboxes = process_image(img)
        bboxes_list = [utils.reformat_bbox_coord(bbox, img.shape[0]) for bbox in rbboxes]

        # Let the user choose which face to follow
        _, bboxes_list, names_list = self.visualizer.plt_img(img, bboxes_list, callback=True)
        print(bboxes_list, names_list)
        # bboxes_list = [[145, 200, 60, 60]]

        # Run the tracking process
        self.track_all(bboxes_list)

    def track_all(self, init_bbox_list):
        with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
            model = Model(sess)
            for init_bbox in init_bbox_list:
                tracker = Tracker(model)
                tracker.initialize(self.s_frames[1], init_bbox)
                self.trackers_list.append(tracker)

            frame_nb = 1
            while frame_nb < len(self.s_frames):
                last_frame = min(frame_nb + config.checking_treshold, len(self.s_frames) - 1)

                for tracker in self.trackers_list:
                    bboxes_list, landmarks = self.track(tracker, frame_nb, last_frame)

                frame_nb = last_frame
                self.verify()
            return

    def track(self, tracker, first_frame=1, last_frame=1):
        """
        Tracks a single face from first_frame to last_frame
        :param tracker:
        :param first_frame:
        :param last_frame:
        :return:
        """
        landmarks_list = []
        bbox_list = []
        with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
            for idx in range(first_frame, last_frame):
                tracker.idx = idx
                bbox, cur_frame = tracker.track(self.s_frames[idx])
                bbox_list.append(bbox)
                cur_frame = cur_frame * 255

                # img_cropped_fd, crop_coord_fd = utils.crop_roi(cur_frame, bbox, 1.4)
                # face_rot, angle = utils.rotate_roi(img_cropped_fd, bbox, cur_frame.shape[0])
                # landmarks = self.fa.get_landmarks(face_rot)
                # if landmarks is not None :
                #     landmarks = utils.landmarks_img_coord(utils.rotate_landmarks(landmarks[-1], face_rot, -angle), crop_coord_fd)
                # landmarks_list.append(landmarks)

                landmarks = None

                # visualization
                self.visualizer.plt_img(cur_frame, [bbox], landmarks=landmarks)

        return bbox_list, landmarks_list

    def verify(self, bbox_list):
        for idx1, bbox in bbox_list:
            for idx2, bbox2 in bbox_list:
                if idx1 != idx2 and utils.bb_intersection_over_union(bbox, bbox2)>0.5:
                    self.correct()
        return True, None

    def correct(self):
        print("Houston, we have a problem")
        return

    def write_bbox(self, bbox):
        self.data_handler.write_data(bbox)

    def check_face(self, img, bbox):
        img_cropped, crop_coord = utils.crop_roi(img, bbox)
        img_rot, angle = utils.rotate_roi(img_cropped, bbox, img.shape[0])
        _, _, rbboxes = process_image(img_rot)

        if len(rbboxes) > 0:
            corrected_bbox = None
            for bbox_fd in rbboxes:
                bbox_fd = utils.reformat_bbox_coord(rbboxes[0], img_cropped.shape[0], img_cropped.shape[1])
                if utils.bb_intersection_over_union(bbox, bbox_fd) > 0.5:
                    return False, None
                else:
                    corrected_bbox = bbox_fd

            return True, corrected_bbox

        return False, None


# Main image processing routine.
def load_seq_video():
    cap = cv2.VideoCapture(config.sequence_path)
    if not os.path.exists(config.img_path):
        os.mkdir(config.img_path)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # Read until video is completed
    frm_count = 0
    while cap.isOpened() and frm_count < 5:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            if not os.path.exists(config.img_path):
                os.makedirs(config.img_path)
            img_write_path = os.path.join(config.img_path, "%05d.jpg" % frm_count)
            if not os.path.exists(img_write_path):
                cv2.imwrite(img_write_path, frame)
            frm_count += 1
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    img_names = sorted(os.listdir(config.img_path))
    s_frames = [os.path.join(config.img_path, img_name) for img_name in img_names]

    return s_frames


def process_image(img, select_threshold=0.35, nms_threshold=0.1):
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


def init():
    # Load the video sequence
    s_frames = load_seq_video()

    # s_frames = ["data/bbox_test/00000.png","data/bbox_test/00001.png"]
    # img = cv2.imread(s_frames[0])
    # b, g, r = cv2.split(img)  # get b,g,r
    # img = cv2.merge([r, g, b])  # switch it to rgb

    # Detect faces in the first image
    img = mpimg.imread(s_frames[0])
    img = np.array(img)

    rclasses, rscores, rbboxes = process_image(img)
    bboxes_list = [utils.reformat_bbox_coord(bbox, img.shape[0]) for bbox in rbboxes]
    print(bboxes_list)


    # Let the user choose which face to follow
    _, bboxes_list = visualization.plt_img(img, bboxes_list, callback=True)
    # bboxes_list = [[145, 200, 60, 60]]

    # Run the tracking process
    for bbox in bboxes_list:
        tracker = MainTracker("b")
        tracker.write_bbox(bbox)
        tracker.run_tracker(bbox, s_frames)


if __name__ == '__main__':
    main_tracker = MainTracker()
    main_tracker.start_tracking()
