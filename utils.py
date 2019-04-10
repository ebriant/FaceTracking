import numpy as np
import config
import cv2
import imutils
import os


def load_seq_video():
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


def get_bbox_dict_ang_pos(bbox_dict, img_shape):
    angles_dict = {}
    for name, data in bbox_dict.items():
        bbox = data[config.BBOX_KEY]
        angles_dict[name] = get_bbox_angular_pos(bbox, img_shape)
    return angles_dict


def get_bbox_angular_pos(bbox, img_shape):
    x_c, y_c = img_shape[0] // 2, img_shape[1] // 2
    x = bbox[0] + bbox[2] - x_c
    y = img_shape[1] - (bbox[1] + bbox[3]) - y_c
    return np.degrees(np.arctan2(y, x))


def get_angular_dist(bbox1, bbox2, img_shape):
    angles1 = get_bbox_angular_pos(bbox1, img_shape)
    angles2 = get_bbox_angular_pos(bbox2, img_shape)
    dist = abs(angles1-angles2)
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
