# The demo credit belongs to Yi-Ting Chen
import os
import cv2
import sys
import numpy as np
from math import cos, sin
import matplotlib.image as mpimg
import config
import data_handler
import utils
import visualization
from FSA_Net.demo.FSANET_model import *
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

K.set_learning_phase(0)  # make sure its testing mode
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

# load model and weights
img_size = 64
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
img_idx = 0
detected = ''  # make this not local variable
time_detection = 0
time_network = 0
time_plot = 0
skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
ad = 0.6

# Parameters
num_capsule = 3
dim_capsule = 16
routings = 2
stage_num = [3, 3, 3]
lambda_d = 1
num_classes = 3
image_size = 64
num_primcaps = 7 * 3
m_dim = 5
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

num_primcaps = 8 * 8 * 3
S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

print('Loading models ...')

configTf = tf.ConfigProto()
configTf.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
configTf.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=configTf)
set_session(sess)  # set this TensorFlow session as the default session for Keras


weight_file1 = 'FSA_Net/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
model1.load_weights(weight_file1)
print('Finished loading model 1.')

weight_file2 = 'FSA_Net/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
model2.load_weights(weight_file2)
print('Finished loading model 2.')

weight_file3 = 'FSA_Net/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
model3.load_weights(weight_file3)
print('Finished loading model 3.')

inputs = Input(shape=(64, 64, 3))
x1 = model1(inputs)  # 1x1
x2 = model2(inputs)  # var
x3 = model3(inputs)  # w/o
avg_model = Average()([x1, x2, x3])

model = Model(inputs=inputs, outputs=avg_model)

print('Start detecting pose ...')
detected_pre = []

def main():
    visualizer = visualization.VisualizerOpencv()
    position_data = data_handler.get_data(os.path.join(config.out_dir, "171214_1.txt"))
    s_frames = utils.get_video_frames()

    for name, data2 in position_data.items():
        data = position_data["d"]
        for idx, bbox in enumerate(data[config.BBOX_KEY]):
            frame = mpimg.imread(s_frames[idx])
            frame = np.array(frame)
            utils.crop_roi(bbox, frame, 2)

            yaw, pitch, roll = get_head_pose(frame, bbox)

            print(yaw, pitch, roll)

            visualizer.prepare_img(frame, idx)
            frame = visualizer.draw_axis(yaw, pitch, roll, tdx=bbox[0] + bbox[2] / 2,
                                    tdy=bbox[1] + bbox[3] / 2, size=min(bbox[2], bbox[3]) / 2)
            visualizer.plt_img()
            visualizer.save_img("data/head_pose_test2")


def get_head_pose(input_img, bbox):

    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    faces = np.empty((1, img_size, img_size, 3))
    faces[0, :, :, :] = cv2.resize(input_img[y1:y2 + 1, x1:x2 + 1, :], (img_size, img_size))
    faces[0, :, :, :] = cv2.normalize(faces[0, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    face = np.expand_dims(faces[0, :, :, :], axis=0)
    p_result = model.predict(face)
    return p_result[0][0], p_result[0][1], p_result[0][2]


if __name__ == '__main__':
    main()
