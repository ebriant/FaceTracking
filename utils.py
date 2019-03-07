import numpy as np
import config
import cv2
import imutils


############## Bbox coord must be in Pixels and on the form (TL corner x, TL corner Y, Width, Height)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

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


def crop_roi(img, bbox, scale=config.checking_scale):
    xmin = max(int(bbox[1] - max(((scale-1)/2) * bbox[3], config.checking_min_size)), 0)
    ymin = max(int(bbox[0] - max(((scale-1)/2) * bbox[2], config.checking_min_size)), 0)
    xmax = min(int(bbox[1] + max(((scale+1)/2) * bbox[3], config.checking_min_size)), img.shape[1])
    ymax = min(int(bbox[0] + max(((scale+1)/2) * bbox[2], config.checking_min_size)), img.shape[0])
    return img[xmin:xmax, ymin:ymax], [xmin, xmax, ymin, ymax]


def bbox_img_coord(bbox, crop_coord):
    height2 = crop_coord[3] - crop_coord[2]
    width2 = crop_coord[1] - crop_coord[0]
    xmin = int((bbox[0] * width2) + crop_coord[0])
    ymin = int((bbox[1] * height2) + crop_coord[2])
    xmax = int((bbox[2] * width2) + crop_coord[0])
    ymax = int((bbox[3] * height2) + crop_coord[2])
    return [xmin, ymin, xmax, ymax]


def rotate_face(img, bbox, img_size):
    """
    Rotate a bounding box depending on it's position in the image
    :param img:
    :param bbox:
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    y, x = bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2
    img_center = (w/2, h/2)

    if y > x and x+y < img_size:
        angle = 90
        # M = cv2.getRotationMatrix2D(img_center, angle, 1)
        # dst = cv2.warpAffine(img, M, (h, w))
        dst = imutils.rotate_bound(img, -angle)

    elif y < x and x+y < img_size:
        angle = 180
        # M = cv2.getRotationMatrix2D(img_center, angle, 1)
        # dst = cv2.warpAffine(img, M, (w, h))
        dst = imutils.rotate_bound(img, -angle)

    elif y < x and x+y > img_size:
        angle = -90
        # M = cv2.getRotationMatrix2D(img_center, -angle, 1)
        # dst = cv2.warpAffine(img, M, (h, w))
        dst = imutils.rotate_bound(img, -angle)

    else:
        angle = 0
        dst = img

    return dst, angle
