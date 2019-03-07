import os
import cv2
import config
import tensorflow as tf
import time
import numpy as np
import utils

from PIL import Image
import matplotlib.image as mpimg
import visualization
# import faceAlignment.face_alignment as f_a
from faceAlignment.face_alignment.api import FaceAlignment, LandmarksType

from PyramidBox.preprocessing import ssd_vgg_preprocessing
from PyramidBox.nets.ssd import g_ssd_model
import PyramidBox.nets.np_methods as np_methods
from MemTrack.tracking.tracker import Tracker, Model

fa = FaceAlignment(LandmarksType._3D, device='cuda:0', flip_input=True)

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


def run_tracker(init_bbox, s_frames, first_frame=0, keep_tracker=None):
    bbox = []
    last_frame = min(first_frame + config.checking_treshold, len(s_frames) - 1)

    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        if keep_tracker is None:
            model = Model(sess)
            tracker = Tracker(model)
            tracker.initialize(s_frames[first_frame], init_bbox)
        else:
            tracker = keep_tracker

        for idx in range(first_frame + 1, last_frame):
            tracker.idx = idx
            bbox, cur_frame = tracker.track(s_frames[idx])
            visualization.plt_img(cur_frame, [bbox])

        img = mpimg.imread(s_frames[last_frame])
        img = np.array(img)
        check, new_bbox = check_tracking(img, bbox)
        print("[INFO] Check result on f%d: %s" % (last_frame, check))

        if check:
            run_tracker(new_bbox, s_frames, last_frame)
            print("[INFO] Bbox corrected from %s to %s" % (bbox, new_bbox))
        else:
            if last_frame < len(s_frames):
                run_tracker(init_bbox, s_frames, last_frame, tracker)


def check_tracking(img, bbox):
    img_cropped, crop_coord = utils.crop_roi(img, bbox)
    rclasses, rscores, rbboxes = process_image(img_cropped)
    if len(rbboxes) > 0:
        bbox_fd = utils.reformat_bbox_coord(rbboxes[0], img_cropped.shape[0], img_cropped.shape[1])
        bbox_fd = utils.bbox_img_coord(bbox_fd, crop_coord)
        # visualization.plt_img(img, [bbox, bbox_fd])
        img_cropped_fd, crop_coord_fd = utils.crop_roi(img, bbox_fd, 1.4)
        face_rot, angle = utils.rotate_face(img_cropped_fd, bbox_fd, img.shape[0])
        preds = fa.get_landmarks(face_rot)[-1]
        visualization.plot_facial_features(face_rot, preds)
        if utils.bb_intersection_over_union(bbox, bbox_fd) < 0.5:
            return True, bbox_fd
    return False, None


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



if __name__ == '__main__':
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
    bbox = visualization.plt_img(img, bboxes_list, rclasses, rscores, callback=True)
    # bboxes_list = [[145, 200, 60, 60]]

    # Run the tracking process
    run_tracker(bbox, s_frames)
