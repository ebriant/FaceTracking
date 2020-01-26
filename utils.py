import numpy as np
import config
import cv2
import imutils
import os
import scipy.io as sio
import torch
from math import cos, sin


def get_video_frames():
    cap = cv2.VideoCapture(config.video_path)
    _, video_name = os.path.split(config.video_path)
    img_dir_path = os.path.join(config.img_dir, video_name[:-4])
    if not os.path.exists(img_dir_path):
        os.mkdir(img_dir_path)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # Read until video is completed
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(os.listdir(img_dir_path)):
        frm_count = 0
        while cap.isOpened() and frm_count < config.max_frame:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Display the resulting frame
                img_write_path = os.path.join(img_dir_path, "%05d.jpg" % frm_count)
                if not os.path.exists(img_write_path):
                    cv2.imwrite(img_write_path, frame)
                frm_count += 1
            # Break the loop
            else:
                break

    # When everything done, release the video capture object
    cap.release()

    img_names = sorted(os.listdir(img_dir_path))
    s_frames = [os.path.join(img_dir_path, img_name) for img_name in img_names[:min(config.max_frame, len(img_names))]]

    return s_frames


# Bbox coord must be in Pixels and on the form (TL corner x, TL corner Y, Width, Height)

def reformat_bbox_coord(bbox, img_width, img_height=0):
    if img_height == 0:
        img_height = img_width
    result = [bbox[1] * img_width, bbox[0] * img_height, (bbox[3] - bbox[1]) * img_width,
              (bbox[2] - bbox[0]) * img_height]
    result = [int(i) for i in result]
    return result


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


def bb_contained(bbox1, bbox2):
    x1_1, y1_1 = bbox1[0], bbox1[1]
    x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x1_2, y1_2 = bbox2[0], bbox2[1]
    x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    return (x1_1 < x1_2 and y1_1 < y1_2 and x2_1 > x2_2 and y2_1 > y2_2) \
           or (x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2)


def bbox_in_roi(bbox1, bbox2, img):
    """
    Check if bbox2 is in bbox1 Region of Interest
    :param bbox1:
    :param bbox2:
    :param img:
    :return:
    """
    xmin, ymin, xmax, ymax = get_roi(bbox1, img)
    x, y = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2
    return xmin < x < xmax and ymin < y < ymax


def get_bbox_dist(bbox1, bbox2):
    x1, y1 = bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2
    x2, y2 = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def resize_bbox(bbox, img, scale=1, make_square=False):
    x_c = bbox[0] + bbox[2] / 2
    y_c = bbox[1] + bbox[3] / 2

    width = bbox[2] * scale
    height = bbox[3] * scale

    if make_square:
        width, height = max(width, height), max(width, height)

    xmin = int(max(0, x_c - width / 2))
    ymin = int(max(0, y_c - height / 2))
    xmax = int(min(img.shape[0], x_c + width / 2))
    ymax = int(min(img.shape[1], y_c + height / 2))

    return [xmin, ymin, xmax - xmin, ymax - ymin]


def get_roi(bbox, img, scale=config.roi_ratio):
    size = max(bbox[2], bbox[3])
    x_c = bbox[0] + bbox[2] / 2
    y_c = bbox[1] + bbox[3] / 2
    new_size = max(config.roi_min_size, size * scale)
    xmin = int(max(0, x_c - new_size / 2))
    ymin = int(max(0, y_c - new_size / 2))
    xmax = int(min(img.shape[0], x_c + new_size / 2))
    ymax = int(min(img.shape[1], y_c + new_size / 2))
    return xmin, ymin, xmax, ymax


def crop_roi(bbox, img, scale=config.roi_ratio):
    xmin, ymin, xmax, ymax = get_roi(bbox, img, scale)
    return img[ymin:ymax, xmin:xmax], [xmin, ymin, xmin - xmax, ymin - ymax]


def bbox_img_coord(bbox, crop_coord):
    return [bbox[0] + crop_coord[0], bbox[1] + crop_coord[1], bbox[2], bbox[3]]


def landmarks_img_coord(landmarks, crop_coord):
    for lm in landmarks:
        lm[0], lm[1] = lm[0] + crop_coord[0], lm[1] + crop_coord[1]
    return landmarks


def rotate_roi(roi, bbox, img_size):
    """
    Rotate a bounding box depending on it's position in the image
    Rotation angle is given in degree counter-clockwise
    :param roi:
    :param bbox:
    :return:
    """

    x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

    if y > x and x + y < img_size:
        angle = 90
        dst = imutils.rotate_bound(roi, -angle)

    elif y < x and x + y < img_size:
        angle = 180
        dst = imutils.rotate_bound(roi, -angle)

    elif y < x and x + y > img_size:
        angle = -90
        dst = imutils.rotate_bound(roi, -angle)
    else:
        angle = 0
        dst = roi

    return dst, angle


def rotate_bbox(bbox, img, angle):
    '''
    Rotate the coordonates of a bounding box
    :param bbox:
    :param img:
    :param angle: 90, 180, -90
    :return:
    '''
    angle = angle % 360
    height, width = img.shape[0], img.shape[1]
    if angle == 0:
        return bbox
    elif angle == 90:
        return [bbox[1], width - (bbox[0] + bbox[2]), bbox[3], bbox[2]]
    elif angle == 180:
        return [width - (bbox[0] + bbox[2]), height - (bbox[1] + bbox[3]), bbox[2], bbox[3]]
    elif angle == 270:
        return [height - (bbox[1] + bbox[3]), bbox[0], bbox[3], bbox[2]]
    else:
        raise Exception("angle not conform")


def rotate_landmarks(landmarks, img, angle):
    angle = angle % 360
    height, width = img.shape[0], img.shape[1]
    for lm in landmarks:
        if angle == 90:
            lm[0], lm[1] = lm[1], width - lm[0]
        if angle == 180:
            lm[0], lm[1] = width - lm[0], height - lm[1]
        if angle == 270:
            lm[0], lm[1] = height - lm[1], lm[0]
    return landmarks


def get_bbox_dict_ang_pos(bbox_dict, img_shape):
    angles_dict = {}
    for name, data in bbox_dict.items():
        bbox = data[config.BBOX_KEY]
        angles_dict[name] = get_angle(bbox, img_shape)
    return angles_dict


def get_angle(bbox, img_shape):
    x_c, y_c = img_shape[0] // 2, img_shape[1] // 2
    x = bbox[0] + bbox[2] - x_c
    y = img_shape[1] - (bbox[1] + bbox[3]) - y_c
    return np.degrees(np.arctan2(y, x))


def get_angular_dist(bbox1, bbox2, img_shape):
    angles1 = get_angle(bbox1, img_shape)
    angles2 = get_angle(bbox2, img_shape)
    dist = abs(angles1 - angles2)
    return dist


def get_list_disorder(list1=[], list2=[]):
    l = len(list1)
    start = list2.index(list1[0])
    list2 = list2[start:] + list2[:start]
    for i, n in enumerate(list1):
        continue
    return


def is_point_in_bbox(bbox, point):
    return bbox[0] < point[0] < bbox[0] + bbox[2] and bbox[1] < point[1] < bbox[1] + bbox[3]


def is_point_in_bbox_list(bbox_list, point):
    for bbox in bbox_list:
        if is_point_in_bbox(bbox, point):
            return True
    return False


def is_between(start, end, mid):
    end = end - start + 360 if (end - start) < 0 else end - start
    mid = mid - start + 360 if (mid - start) < 0 else mid - start
    return 0 < mid < end


def is_bbox_in_bbox_list(bbox_list, bbox, th):
    for idx, bbox2 in enumerate(bbox_list):
        if bb_intersection_over_union(bbox2, bbox) > th:
            return True, bbox2
    return False, None


def grad(v1, v2, t):
    return np.subtract(np.array(v1), np.array(v2)) / t


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights
