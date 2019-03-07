import cv2
import numpy as np
import os
import config
import random


def draw_bbox(img, bbox, label="", color=(0, 255, 0), thickness=2):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    cv2.rectangle(img, p1, p2, color, thickness)
    p1 = (p1[0], p1[1] - 10)
    cv2.putText(img, str(label), p1, cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return


def plot_facial_features(img, features_list):
    b, g, r = cv2.split(img)  # get b,g,r
    img = cv2.merge([r, g, b])  # switch it to rgb
    for i in range(0, 68):
        cv2.circle(img, (features_list[i, 0], features_list[i, 1]), 2, color=(0, 0, 255))
    cv2.imshow("%.2f" % (random.random()), img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plt_img(img, bboxes_list, classes=[], scores=[], title="image", callback=False, color=(0, 255, 0)):
    b, g, r = cv2.split(img)  # get b,g,r
    img = cv2.merge([r, g, b])  # switch it to rgb
    selected_bbox = []
    if np.amax(img) <= 1:
        img = img * 255
    img = np.array(img, dtype=np.uint8)

    for bbox in bboxes_list:
        draw_bbox(img, bbox, color=color)
        # bbox = [xmin, ymin, xmax, ymax]
        # bboxes_list_px.append(bbox)
        # draw_bbox(img, bbox, color=color)

    def mouse_position(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for bbox in bboxes_list:
                if is_in_bbox(bbox, x, y):
                    draw_bbox(img, bbox, "selected", (0, 0, 200))
                    cv2.imshow(title, img)
                    selected_bbox.append(bbox)

    def is_in_bbox(box, x, y):
        if box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]:
            return True
        return False

    # Save img in the output folder
    img_names = sorted(os.listdir(config.out_folder))
    if len(img_names) == 0:
        name = 0
    else:
        name = int(img_names[-1][:5]) + 1
    img_write_path = os.path.join(config.out_folder, "%05d.png" % name)
    cv2.imwrite(img_write_path, img)

    # Shows image and wait for user action if callback
    cv2.namedWindow(title)
    if callback:
        cv2.setMouseCallback(title, mouse_position)
        cv2.imshow(title, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return selected_bbox[0]
    else:
        while True:
            cv2.imshow(title, img)
            if cv2.waitKey(1):
                break
    return
