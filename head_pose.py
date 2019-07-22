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

visualizer = visualization.VisualizerOpencv()
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

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))
#
# txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

frame_num = 1

position_data = data_handler.get_data(os.path.join(config.out_dir, "171214_1.txt"))
s_frames = utils.load_seq_video()

for name, data2 in position_data.items():
    data = position_data["c"]
    for idx, bbox in enumerate(data[config.BBOX_KEY]):
        frame = mpimg.imread(s_frames[idx])
        frame = np.array(frame)

        utils.crop_roi(bbox, frame, 2)

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
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        # Crop image
        img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

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


        print(yaw_predicted, pitch_predicted, roll_predicted)

        # print( new frame with cube and axis)
        # txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))

        # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
        frame = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                        tdy=(y_min + y_max) / 2, size=bbox_height / 2)

        visualizer.prepare_img(frame, idx)
        visualizer.plt_img()
        visualizer.save_img("data/head_pose_test")

def get_head_pose():
    return
