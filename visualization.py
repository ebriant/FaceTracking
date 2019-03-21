import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import config
import random

fig, ax = plt.subplots(1)


class VisualizerPlt:
    def __init__(self):
        self.fig, self.ax = plt.subplot()
        self.rectangles_list = []

    def draw_bbox(self, ax, bbox, label="", color=(0, 255, 0), thickness=2):
        rect = patches.Rectangle((bbox[0], bbox[1] + bbox[3]), bbox[2], bbox[3], linewidth=1, edgecolor=color)
        self.ax.add_patch(rect)

        return

    def plt_img(img, bboxes_list, *, landmarks=None, title="image", callback=False, color=(0, 1, 0)):
        ax.clear()
        selected_bbox = []
        # Display the image
        ax.imshow(img)
        for bbox in bboxes_list:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        def is_in_bbox(box, x, y):
            if box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]:
                return True
            return False

        def onclick(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            for bbox in bboxes_list:
                if is_in_bbox(bbox, ix, iy):
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                             linewidth=1, edgecolor=config.SELECTED_COLOR, facecolor='none')
                    ax.add_patch(rect)
                    plt.draw()
                    selected_bbox.append(bbox)
            return

        def press(event):
            if event.key == 'x' or event.key == ' ':
                plt.close()
                return

        if callback:
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            cid2 = fig.canvas.mpl_connect('key_press_event', press)
        plt.tight_layout()
        plt.show()
        return img, selected_bbox


class VisualizerOpencv:
    def __init__(self):
        self.img = None
        self.BBOX_COLOR = tuple([int(a * 255) for a in config.BBOX_COLOR])
        self.SELECTED_COLOR = tuple([int(a * 255) for a in config.SELECTED_COLOR])
        self.idx = 0

    def draw_bbox(self, bbox, label="", color=(0, 255, 0), thickness=2):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(self.img, p1, p2, color, thickness)
        p1 = (p1[0], p1[1] - 10)
        cv2.putText(self.img, str(label), p1, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
        return

    def plot_facial_features(self, landmarks_list):
        for i in range(0, 68):
            cv2.circle(self.img, (int(landmarks_list[i, 0]), int(landmarks_list[i, 1])), 1, color=(0, 0, 255))

    def select_bbox(self, img, bboxes_list, *, title="image"):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        selected_bbox = []
        names_list = []
        self.img = np.array(self.img, dtype=np.uint8)
        frame_name = str("Frame %d" % self.idx)
        cv2.putText(self.img, frame_name, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        self.idx = self.idx + 1

        for bbox in bboxes_list:
            self.draw_bbox(bbox, color=self.BBOX_COLOR)
            # self.draw_bbox(data[config.BBOX_KEY], label=name, color=color)

        def mouse_position(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for bbox in bboxes_list:
                    if is_in_bbox(bbox, x, y):
                        self.draw_bbox(bbox, label="selected", color=self.SELECTED_COLOR)
                        cv2.imshow(title, self.img)
                        selected_bbox.append(bbox)
                        name = input("Enter a name for the selected children: ")
                        names_list.append(name)

        def is_in_bbox(box, x, y):
            if box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]:
                return True
            return False

        # Shows image and wait for user action if callback
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return self.img, selected_bbox, names_list

    def plt_img(self, img, tracking_data, title="image"):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img, dtype=np.uint8)
        frame_name = str("Frame %d" % self.idx)
        cv2.putText(self.img, frame_name, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        self.idx = self.idx + 1

        for name, data in tracking_data.items():
            self.draw_bbox(data[config.BBOX_KEY], label=name, color=self.BBOX_COLOR)

            if config.LANDMARKS_KEY in data:
                self.plot_facial_features(data[config.LANDMARKS_KEY])

        # # Save img in the output folder
        # img_names = sorted(os.listdir(config.out_folder))
        # if len(img_names) == 0:
        #     name = 0
        # else:
        #     name = int(img_names[-1][:5]) + 1
        # img_write_path = os.path.join(config.out_folder, "%05d.png" % name)
        # cv2.imwrite(img_write_path, img)

        # Shows image and wait for user action if callback
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow(title, self.img)
            if cv2.waitKey(1):
                break
        return self.img
