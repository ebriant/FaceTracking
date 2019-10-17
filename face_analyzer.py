import data_handler
import os
import config
import numpy as np
from faceAlignment.face_alignment.api import FaceAlignment, LandmarksType
import face_alignment
import utils
import matplotlib.image as mpimg
import visualization
import copy

class FaceAnalizer:
    def __init__(self):
        print(os.path.join(config.out_dir, "171214_1.txt"))
        self.visualizer = visualization.ImageProcessor()
        self.position_data = data_handler.get_data(os.path.join(config.out_dir, "171214_1.txt"))
        self.face_aligner = FaceAlignment(LandmarksType._3D, device='cuda:0', flip_input=True)
        self.positions = {}
        self.s_frames = utils.get_video_frames()
        self.data = {name: [] for name in self.position_data}
        self.cur_img = None

    def get_all_positions(self):
        frame_idx = 0
        while frame_idx < len(self.s_frames):
            last_frame = min(frame_idx + 1, len(self.s_frames))
            for name, data in self.position_data.items():
                for frame in range(frame_idx, last_frame):
                    self.get_position(name, data[config.BBOX_KEY], frame)
            frame_idx = last_frame

        # for name, data in self.position_data.items():
        #     name, data = "c", self.position_data["c"]
        #     self.get_position(name, data[config.BBOX_KEY])

        # data_handler.save_data(None, self.data)

    def get_position(self, name, data, frame):
        self.cur_img = mpimg.imread(self.s_frames[frame])
        self.cur_img = np.array(self.cur_img)
        face, crop_coord = utils.crop_roi(data[frame], self.cur_img, 2)
        face, angle = utils.rotate_roi(face, data[frame], self.cur_img.shape[0])

        landmarks = self.face_aligner.get_landmarks(face)
        if landmarks is None:
            self.data[name].append(None)
        else:
            landmarks = np.array(landmarks)[0]
            # lm2 = copy.deepcopy(landmarks)
            # utils.rotate_landmarks(landmarks, face, -angle)
            # utils.landmarks_img_coord(landmarks, crop_coord)
            #
            # self.visualizer.prepare_img(self.cur_img)
            # self.visualizer.plot_facial_features(landmarks, size=2)
            # self.visualizer.resize(1)
            # self.visualizer.plt_img({})

            # self.visualizer.prepare_img(face)
            # self.visualizer.plot_facial_features(lm2, size=2)
            # self.visualizer.resize(1)
            # self.visualizer.plt_img({})

            orientation = list(face_alignment.face_orientation(landmarks[43:48], landmarks[36:42], landmarks[8]))

            print(orientation)
            print(str(frame) + "_" + name)
            self.visualizer.prepare_img(face, cvt_color=True)
            self.visualizer.plot_facial_features(landmarks)
            # self.visualizer.draw_axis(orientation[1], orientation[0], orientation[2], size=20)
            self.visualizer.save_img("data/head_pose_landmark_test2", str(frame) + "_" + name + ".jpg")
            self.data[name].append(orientation)
            # if self.check_grad(orientation, name, frame):
            #
            # else:
            #     self.data[name].append(None)
        return

    def check_grad(self, vector, name, frame):
        if len(self.data[name]) == 0:
            return True
        last_vect, idx = self.get_last(name)
        gradient = np.linalg.norm(utils.grad(vector, last_vect, frame - idx))
        return gradient < config.face_orientation_max_grad

    def get_last(self, name):
        for i in range(len(self.data[name]) - 1, -1, -1):
            if self.data[name][i] is not None:
                return self.data[name][i], i
        return None

fa = FaceAnalizer()
fa.get_all_positions()
