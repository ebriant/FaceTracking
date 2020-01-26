import cv2
import utils
import data_handler
import config
import visualization
import os
import math

MAX_FRAME = 5000
OVERWRITE = True

s_frame = utils.get_video_frames()
data = data_handler.get_data("data/171214_1.txt")
visualizer = visualization.ImageProcessor()

for i in range(0, MAX_FRAME):

    if not OVERWRITE and os.path.isfile(s_frame[i]):
        continue
    print(i)

    visualizer.open_img_path(s_frame[i], i)
    for name, labels in data.items():
        bbox = labels[config.BBOX_KEY][i]
        visualizer.draw_bbox(bbox)

        # yaw = labels[config.YAW_KEY][i] * 180 / math.pi
        # pitch = labels[config.PITCH_KEY][i] * 180 / math.pi
        # roll = labels[config.ROLL_KEY][i] * 180 / math.pi
        # visualizer.draw_axis(yaw, pitch, roll, tdx=bbox[0]+bbox[2]/2, tdy=bbox[1]+bbox[3]/2, size=50)

        landmarks = labels[config.LANDMARKS_KEY][i]
        # print(landmarks)

        visualizer.plot_facial_features(landmarks)

    visualizer.save_img("data/output/171214_1_results_landmarks")
