import os
from math import cos, sin
import cv2
import numpy as np
import config


class ImageProcessor:
    def __init__(self, img=None, idx=0, cvt_color=False):
        self.img = None
        if img is not None:
            self.idx = idx
            self.prepare_img(img, idx, cvt_color=cvt_color)
        self.BBOX_COLOR = tuple([int(a * 255) for a in reversed(config.BBOX_COLOR)])
        self.SELECTED_COLOR = tuple([int(a * 255) for a in reversed(config.SELECTED_COLOR)])
        self.size = self.img.shape[0]

    def prepare_img(self, img, frame_idx=None, *, cvt_color=False, scaling=1):
        self.idx = frame_idx
        if cvt_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = np.array(img, dtype=np.uint8)
        if scaling != 1:
            self.img = cv2.resize(self.img, None, fx=scaling, fy=scaling)
        if frame_idx is not None:
            cv2.putText(self.img, "Frame %d" % self.idx, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

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
        p1 = (int(bbox[0] * self.size), int(bbox[1] * self.size))
        p2 = (int(bbox[0] * self.size + bbox[2] * self.size), int(bbox[1] * self.size + bbox[3] * self.size))
        cv2.rectangle(self.img, p1, p2, color, **kwargs)
        p1 = (p1[0], p1[1] - 15)
        cv2.putText(self.img, str(label), p1, cv2.FONT_HERSHEY_DUPLEX, self.size / 700, color)

    def plt_frame_data(self, frame_data):
        for name, item in frame_data.items():
            self.draw_bbox(item[config.BBOX_KEY], name, color=self.BBOX_COLOR, thickness=2)

    def select_bbox(self, bboxes_list, *, title="image"):
        selected_bbox = []
        names_list = []
        self.tmp_bbox = []
        for bbox in bboxes_list:
            self.draw_bbox(bbox, color=self.BBOX_COLOR)
            # self.draw_bbox(data[config.BBOX_KEY], label=name, color=color)

        def mouse_position(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                for bbox in bboxes_list:
                    if is_in_bbox(bbox, x, y):
                        name = input("Enter a name for the selected children: ")
                        names_list.append(name)
                        self.draw_bbox(bbox, label=name, color=self.SELECTED_COLOR)
                        cv2.imshow(title, self.img)
                        selected_bbox.append(bbox)
            if event == cv2.EVENT_RBUTTONDOWN:
                print(x, y)
                self.tmp_bbox.append(x)
                self.tmp_bbox.append(y)
            if event == cv2.EVENT_RBUTTONUP:
                print(x, y, self.tmp_bbox)
                tmp_bbox = [min(self.tmp_bbox[0], x), min(self.tmp_bbox[1], y), abs(self.tmp_bbox[0] - x),
                            abs(self.tmp_bbox[1] - y)]
                print(tmp_bbox)
                bboxes_list.append(tmp_bbox)
                self.draw_bbox(tmp_bbox, color=self.BBOX_COLOR)
                cv2.imshow(title, self.img)
                self.tmp_bbox = []

        def is_in_bbox(box, x, y):
            if box[0] <= x / self.size <= box[0] + box[2] and box[1] <= y / self.size <= box[1] + box[3]:
                return True
            return False

        # Shows image and wait for user action if callback
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return self.img, selected_bbox, names_list

    def show(self):
        title = "img"
        cv2.namedWindow(title)
        while True:
            cv2.imshow(title, self.img)
            if cv2.waitKey(1):
                break

    def get_img(self):
        return self.img

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
                bbox = [x/self.size for x in bbox]
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

    def blur_area(self, bbox, ksize=20):
        crop_box = [int(bbox[1] * self.size), int((bbox[1] + bbox[3]) * self.size),
                    int(bbox[0]), int((bbox[0] + bbox[2]) * self.size)]
        sub_face = self.img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face, (ksize, ksize), 30)
        # merge this blurry rectangle to our final image
        self.img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]] = sub_face
