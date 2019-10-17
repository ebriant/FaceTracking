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

visualizer = visualization.ImageProcessor()
cudnn.enabled = True
batch_size = 1
gpu = 0
out_dir = 'output/video'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

print('Loading snapshot.')
# Load snapshot
saved_state_dict = torch.load("data/head_pose_snapshot/hopenet_robust_alpha1.pkl")
model.load_state_dict(saved_state_dict)

print('Loading data.')

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model.cuda(gpu)

print('Ready to test network.')

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)


def get_head_pose(img, bbox):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[0] + bbox[2]
    y_max = bbox[1] + bbox[3]

    bbox_width = abs(x_max - x_min)
    bbox_height = abs(y_max - y_min)
    x_min -= 2 * bbox_width / 4
    x_max += 2 * bbox_width / 4
    y_min -= 3 * bbox_height / 4
    y_max += bbox_height / 4
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(img.shape[1], x_max)
    y_max = min(img.shape[0], y_max)
    # Crop image
    img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

    img = Image.fromarray(img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(gpu)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted


if __name__ == '__main__':
    position_data = data_handler.get_data(os.path.join(config.out_dir, "171214_1.txt"))
    s_frames = utils.get_video_frames()

    with open(os.path.join(config.label_dir, "171214_1_bbox.txt")) as f:
        labels = eval(f.read())

    for frame_idx, frame_label in labels.items():
        visualizer.open_img_path(s_frames[frame_idx])
        frame = visualizer.img
        for name, label in frame_label.items():
            if label[2] > 0 and label[3] > 0:
                bbox = label
                face, coord = utils.crop_roi(bbox, frame, 2)
                bbox2 = [bbox[0]-coord[0], bbox[1]-coord[1], bbox[2], bbox[3]]
                visualizer.prepare_img(face)
                # frame2, angle = utils.rotate_roi(frame, bbox, frame.size)
                # bbox2 = utils.rotate_bbox(bbox, frame, angle)
                # yaw_predicted, pitch_predicted, roll_predicted = get_head_pose(frame, bbox)

                yaw, pitch, roll = get_head_pose(face, bbox2)
                # roll -= angle

                visualizer.draw_axis(yaw, pitch, roll, tdx = bbox2[0] + bbox2[2] / 2,
                                             tdy=bbox2[1] + bbox2[3] / 2, size=min(bbox2[2], bbox2[3]) / 2)

                visualizer.plt_img()
                visualizer.save_img("data/head_pose_test2", str(frame_idx)+ "_" + name +".jpg")

# for name, data2 in position_data.items():
#     data = position_data["d"]
#     for idx, bbox in enumerate(data[config.BBOX_KEY]):
#         frame = mpimg.imread(s_frames[idx])
#         frame = np.array(frame)
#
#         utils.crop_roi(bbox, frame, 2)
#
#         yaw_predicted, pitch_predicted, roll_predicted = get_head_pose(frame, bbox)
#
#         print(yaw_predicted, pitch_predicted, roll_predicted)
#
#         frame = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=bbox[0] + bbox[2] / 2,
#                         tdy=bbox[1] + bbox[3] / 2, size=bbox[3] / 2)
#         visualizer.prepare_img(frame, idx)
#         visualizer.plt_img()
#         visualizer.save_img("data/head_pose_test")

