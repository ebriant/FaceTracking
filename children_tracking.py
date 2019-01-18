import os
import cv2
import config
import tensorflow as tf
import time
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import visualization

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
    while cap.isOpened() and frm_count < 3000:
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
    bbox_corner_dims = []
    last_frame= min(first_frame+config.checking_treshold,len(s_frames))
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        if keep_tracker is None:
            model = Model(sess)
            tracker = Tracker(model)
            tracker.initialize(s_frames[first_frame], init_bbox)
        else:
            tracker = keep_tracker

        for idx in range(first_frame+1, last_frame):
            tracker.idx = idx
            bbox_corner_dims, cur_frame = tracker.track(s_frames[idx])
            bbox = np.array([bbox_corner_dims[1], bbox_corner_dims[0],
                             bbox_corner_dims[1] + bbox_corner_dims[3],
                             bbox_corner_dims[0] + bbox_corner_dims[2]])
            visualization.plt_img(cur_frame, np.array([bbox]))

        img = mpimg.imread(s_frames[last_frame])
        img = np.array(img)
        print("..........", bbox)
        check = check_tracking(img, bbox, bbox_corner_dims)
        print("Check result on f%d: %s" %(last_frame, check))
        if isinstance(check, list):
            check = [check[1], check[0], check[3] - check[1], check[2] - check[0]]
            sess.close()
            run_tracker(check, s_frames, last_frame)
        else:
            if last_frame<len(s_frames):
                run_tracker(init_bbox, s_frames, last_frame, tracker)


def crop_img(img, bbox, scale=2):
    xmin = max(int(bbox[1] - ((scale-1)/2) * bbox[3]), 0)
    ymin = max(int(bbox[0] - ((scale-1)/2) * bbox[2]), 0)
    xmax = min(int(bbox[1] + ((scale+1)/2) * bbox[3]), img.shape[1])
    ymax = min(int(bbox[0] + ((scale+1)/2) * bbox[2]), img.shape[0])
    return img[xmin:xmax, ymin:ymax], [xmin, xmax, ymin, ymax]


def bbox_img_coord(bbox, crop_coord):
    height2 = crop_coord[3] - crop_coord[2]
    width2 = crop_coord[1] - crop_coord[0]
    xmin = int((bbox[0] * width2) + crop_coord[0])
    ymin = int((bbox[1] * height2) + crop_coord[2])
    xmax = int((bbox[2] * width2) + crop_coord[0])
    ymax = int((bbox[3] * height2) + crop_coord[2])
    return [xmin, ymin, xmax, ymax]


def check_tracking(img, bbox, bbox_corner_dims):
    img2, crop_coord = crop_img(img, bbox_corner_dims)
    rclasses, rscores, rbboxes = process_image(img2)
    if len(rbboxes)>0:
        bbox2 = bbox_img_coord(rbboxes[0], crop_coord)
        visualization.plt_img(img, np.array([bbox,bbox2]))
        if bb_intersection_over_union(bbox, bbox2) < 0.5:
            return bbox2
    return False


def bb_intersection_over_union(boxA, boxB):
    print("----",boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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



def reformat_bboxes_corner_dimensions(bboxes_list):
    result =[]
    for bbox in bboxes_list:
        result.append([bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]])
    return result


if __name__ == '__main__':
    s_frames = load_seq_video()

    img = mpimg.imread(s_frames[0])
    img = np.array(img)
    rclasses, rscores, rbboxes = process_image(img)
    bboxes_list = visualization.plt_img(img, rbboxes, rclasses, rscores, callback=True)
    bboxes_list = reformat_bboxes_corner_dimensions(bboxes_list)

    # bboxes_list = [[145, 200, 60, 60]]

    run_tracker(bboxes_list[0], s_frames)
