import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import deep_head_pose.code.hopenet as hopenet
import utils
import data_handler
import config
import matplotlib.image as mpimg
import numpy as np
import visualization
import time

MAX_FRAME = 1000
visualizer = visualization.ImageProcessor()

if __name__ == '__main__':
    position_data = data_handler.get_data(os.path.join(config.out_dir, "171214_1.txt"))
    s_frames = utils.get_video_frames()

    # with open(os.path.join(config.label_dir, "171214_1_bbox.txt")) as f:
    #     labels = eval(f.read())

    for name, data in position_data.items():
        for idx in range(MAX_FRAME):
            bbox = data[config.BBOX_KEY][idx]
            img = visualizer.open_img_path(s_frames[idx])
            visualizer.img, coord = utils.crop_roi(bbox, visualizer.img, 2)
            visualizer.img, angle = utils.rotate_roi(visualizer.img, bbox, img.shape[0])
            visualizer.save_img('C:/Users/Horio/Google Drive/Research/Croped_faces', "%05d_%s.jpg" % (idx, name))
