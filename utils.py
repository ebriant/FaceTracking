import numpy as np
import config
import cv2
import imutils


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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


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


def get_roi(bbox, img):
    size = max(bbox[2], bbox[3])
    x_c = bbox[0] + bbox[2] / 2
    y_c = bbox[1] + bbox[3] / 2
    new_size = max(config.checking_min_size, size * config.checking_scale)
    xmin = int(max(0, x_c - new_size / 2))
    ymin = int(max(0, y_c - new_size / 2))
    xmax = int(min(img.shape[0], x_c + new_size / 2))
    ymax = int(min(img.shape[1], y_c + new_size / 2))
    return xmin, ymin, xmax, ymax


def crop_roi(img, bbox, scale=config.checking_scale):
    size = max(bbox[2], bbox[3])
    x_c = bbox[0] + bbox[2] / 2
    y_c = bbox[1] + bbox[3] / 2
    new_size = max(config.checking_min_size, size * scale)
    xmin = int(max(0, x_c - new_size / 2))
    ymin = int(max(0, y_c - new_size / 2))
    xmax = int(min(img.shape[0], x_c + new_size / 2))
    ymax = int(min(img.shape[1], y_c + new_size / 2))

    return img[ymin:ymax, xmin:xmax], [xmin, ymin, xmin - xmax, ymin - ymax]


def bbox_img_coord(bbox, crop_coord):
    return [bbox[0] + crop_coord[0], bbox[1] + crop_coord[1], bbox[2], bbox[3]]


def landmarks_img_coord(landmarks, crop_coord):
    for lm in landmarks:
        lm[0], lm[1] = lm[0] + crop_coord[0], lm[1] + crop_coord[1]
    return landmarks


def rotate_roi(img, bbox, img_size):
    """
    Rotate a bounding box depending on it's position in the image
    Rotation angle is given in degree counter-clockwise
    :param img:
    :param bbox:
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    img_center = (w / 2, h / 2)

    if y > x and x + y < img_size:
        angle = 90
        # M = cv2.getRotationMatrix2D(img_center, angle, 1)
        # dst = cv2.warpAffine(img, M, (h, w))
        dst = imutils.rotate_bound(img, -angle)

    elif y < x and x + y < img_size:
        angle = 180
        # M = cv2.getRotationMatrix2D(img_center, angle, 1)
        # dst = cv2.warpAffine(img, M, (w, h))
        dst = imutils.rotate_bound(img, -angle)

    elif y < x and x + y > img_size:
        angle = -90
        # M = cv2.getRotationMatrix2D(img_center, -angle, 1)
        # dst = cv2.warpAffine(img, M, (h, w))
        dst = imutils.rotate_bound(img, -angle)
    else:
        angle = 0
        dst = img

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


def get_bbox_dict_ang_pos(bbox_dict, img):
    x_c, y_c = img.shape[0] // 2, img.shape[1] // 2
    angles_dict = {}
    for name, data in bbox_dict.items():
        bbox = data[config.BBOX_KEY]
        angles_dict[name] = get_bbox_angular_pos(bbox, img, center_coord=(x_c, y_c))
    return angles_dict


def get_bbox_angular_pos(bbox, img, center_coord=None):
    if center_coord is not None:
        x_c, y_c = center_coord[0], center_coord[1]
    else:
        x_c, y_c = img.shape[0] // 2, img.shape[1] // 2
    x = bbox[0] + bbox[2] - x_c
    y = img.shape[1] - (bbox[1] + bbox[3]) - y_c
    return np.degrees(np.arctan2(y, x))
