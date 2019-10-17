import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import config
import random
from math import cos, sin


class ImageProcessor:
    def __init__(self, img=None):
        self.img = None
        if img is not None:
            self.prepare_img(img)
        self.BBOX_COLOR = tuple([int(a * 255) for a in reversed(config.BBOX_COLOR)])
        self.SELECTED_COLOR = tuple([int(a * 255) for a in reversed(config.SELECTED_COLOR)])
        self.idx = 0

    def prepare_img(self, img, frame_idx=None, *, cvt_color=False, scaling=1):
        self.idx = frame_idx
        if cvt_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = np.array(img, dtype=np.uint8)
        if scaling != 1:
            self.img = cv2.resize(self.img, None, fx=scaling, fy=scaling)
        if frame_idx is not None:
            cv2.putText(self.img, "Frame %d" % self.idx, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def open_img_path(self, img_path, frame_idx=None, scaling=1):
        img = cv2.imread(img_path)
        self.prepare_img(img, frame_idx, cvt_color=False, scaling=scaling)
        return self.img

    def resize(self, scale):
        self.img = cv2.resize(self.img, None, fx=scale, fy=scale)

    def save_img(self, out_dir, name=None):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if name is None:
            name = "%05d.jpg" % self.idx
        img_write_path = os.path.join(out_dir, name)
        cv2.imwrite(img_write_path, self.img)

    def draw_bbox(self, bbox, label="", color=(0, 255, 0), **kwargs):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(self.img, p1, p2, color, **kwargs)
        p1 = (p1[0], p1[1] - 10)
        cv2.putText(self.img, str(label), p1, cv2.FONT_HERSHEY_DUPLEX, 1, color)
        return

    def plot_facial_features(self, landmarks_list, size=1):
        for i in range(0, 68):
            cv2.circle(self.img, (int(landmarks_list[i, 0]), int(landmarks_list[i, 1])), size, color=(0, 0, 255))

    def select_bbox(self, bboxes_list, *, title="image"):
        selected_bbox = []
        names_list = []

        for bbox in bboxes_list:
            self.draw_bbox(bbox, color=self.BBOX_COLOR)
            # self.draw_bbox(data[config.BBOX_KEY], label=name, color=color)

        def mouse_position(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for bbox in bboxes_list:
                    if is_in_bbox(bbox, x, y):
                        name = input("Enter a name for the selected children: ")
                        names_list.append(name)
                        self.draw_bbox(bbox, label=name, color=self.SELECTED_COLOR)
                        cv2.imshow(title, self.img)
                        selected_bbox.append(bbox)

        def is_in_bbox(box, x, y):
            if box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]:
                return True
            return False

        # Shows image and wait for user action if callback
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return self.img, selected_bbox, names_list

    def plt_img(self, tracking_data=None, title="image"):
        if tracking_data is None:
            tracking_data = {}
        for name, data in tracking_data.items():
            bbox = [int(e) for e in data[config.BBOX_KEY]]
            self.draw_bbox(bbox, label=name, color=self.BBOX_COLOR, thickness=2)

            if config.LANDMARKS_KEY in data:
                self.plot_facial_features(data[config.LANDMARKS_KEY])

        # Shows image and wait for user action if callback
        cv2.namedWindow(title)
        while True:
            cv2.imshow(title, self.img)
            if cv2.waitKey(1):
                break
        return self.img

    def ask_ground_truth(self, title="image"):
        centers = []

        def mouse_position(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                centers.append((x, y))
                cv2.circle(self.img, (x, y), 4, color=self.SELECTED_COLOR, thickness=5)
                cv2.imshow(title, self.img)

        # Shows image and wait for user action if callback
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, self.img)
        while True:
            key = cv2.waitKey()
            if key == 8 or key == ord('b'):
                print("reset")
                return None
            else:
                break
        cv2.destroyAllWindows()
        return centers

    def ask_bboxes(self, title="image"):
        bbox_list = []
        tmp_x, tmp_y = 0, 0

        def mouse_position(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global tmp_x, tmp_y
                tmp_x, tmp_y = x, y
                cv2.imshow(title, self.img)
            if event == cv2.EVENT_LBUTTONUP:
                bbox = [min(tmp_x, x), min(tmp_y, y), abs(tmp_x - x), abs(tmp_y - y)]
                bbox_list.append(bbox)
                self.draw_bbox(bbox, thickness=2)

                cv2.imshow(title, self.img)

        # Shows image and wait for user action if callback
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, self.img)
        while True:
            key = cv2.waitKey()
            if key == 8:
                print("reset")
                return None
            elif key == ord('s'):
                print("skip")
                bbox_list.append([0, 0, 0, 0])
            elif key == ord('b'):
                print("back")
                bbox_list.pop()
            else:
                break
        cv2.destroyAllWindows()
        return bbox_list

    def draw_axis(self, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx is None and tdy is None:
            height, width = self.img.shape[:2]
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

        cv2.line(self.img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(self.img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(self.img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    def blur_area(self, bbox, ksize=20):
        sub_face = self.img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face, (ksize, ksize), 30)
        # merge this blurry rectangle to our final image
        self.img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = sub_face

