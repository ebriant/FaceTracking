import cv2
import utils
import data_handler
import config
import visualization
import children_tracker
import os

MAX_FRAME = 10000
OVERWRITE = True

s_frame = utils.get_video_frames()
data = data_handler.get_data("data/output/171214_1.txt")
visualizer = visualization.ImageProcessor()
for i in range(5040, MAX_FRAME):

    if not OVERWRITE and os.path.isfile(s_frame[i]):
        continue
    print(i)
    visualizer.open_img_path(s_frame[i], i)
    _, _, detected_faces = children_tracker.detect_faces(cv2.cvtColor(visualizer.img, cv2.COLOR_BGR2RGB), 0.6)
    detected_faces = [utils.reformat_bbox_coord(bbox, visualizer.img.shape[0], visualizer.img.shape[1]) for bbox in detected_faces]
    for bbox in detected_faces:
        visualizer.blur_area(bbox, 15)
        # visualizer.blur_area(bbox)
    for name, labels in data.items():
        bbox = labels[config.BBOX_KEY][i]
        visualizer.blur_area(utils.resize_bbox(bbox, visualizer.img, 1.2), 15)
        # visualizer.blur_area(bbox)

    # for name, labels in data.items():
    #     bbox = labels[config.BBOX_KEY][i]
    #     visualizer.draw_bbox(bbox, name)

    visualizer.save_img("data/output/171214_1_blured3")
