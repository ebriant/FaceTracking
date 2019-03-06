import numpy as np
import config
import cv2

def bb_intersection_over_union(boxA, boxB):
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


def crop_roi(img, bbox, scale=config.checking_scale):
    xmin = max(int(bbox[1] - max(((scale-1)/2) * bbox[3], config.checking_min_size)), 0)
    ymin = max(int(bbox[0] - max(((scale-1)/2) * bbox[2], config.checking_min_size)), 0)
    xmax = min(int(bbox[1] + max(((scale+1)/2) * bbox[3], config.checking_min_size)), img.shape[1])
    ymax = min(int(bbox[0] + max(((scale+1)/2) * bbox[2], config.checking_min_size)), img.shape[0])
    return img[xmin:xmax, ymin:ymax], [xmin, xmax, ymin, ymax]


def rotate_face(img, bbox, img_size):
    """

    :param img:
    :param bbox:
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    y, x = bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2
    center = (w/2, h/2)
    if y>x and x+y<img_size:
        rot = 90
        M = cv2.getRotationMatrix2D(center, rot, 1)
        dst = cv2.warpAffine(img, M, (h, w))
    elif y<x and x+y<img_size:
        rot = 180
        M = cv2.getRotationMatrix2D(center, rot, 1)
        dst = cv2.warpAffine(img, M, (w, h))
    elif y<x and x+y>img_size:
        rot = -90
        M = cv2.getRotationMatrix2D(center, rot, 1)
        dst = cv2.warpAffine(img, M, (h, w))
    else:
        rot = 0
        dst=img
    return dst, rot
