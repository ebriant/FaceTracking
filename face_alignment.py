import cv2
import numpy as np
import json
import os
import visualization


class FaceAligner:
    def __init__(self, target_height=0.6, target_eye_center=0.35, target_size=(280, 320)):
        self.target_height = target_height
        self.target_eye_center = target_eye_center
        self.target_size = target_size

    def align_face(self, img, l_eye, r_eye, mouth):
        r_eye_center = np.mean(l_eye, axis=0)
        l_eye_center = np.mean(r_eye, axis=0)
        mouth_center = np.mean(mouth, axis=0)
        dy = r_eye_center[1] - l_eye_center[1]
        dx = r_eye_center[0] - l_eye_center[0]

        roll = 90 if dx == 0 else np.degrees(np.arctan(dy / dx))

        eyes_center = np.mean([l_eye_center, r_eye_center], axis=0)

        dist = np.linalg.norm((eyes_center - mouth_center)[:2])
        dist = dist / self.target_size[1]
        scale = self.target_height / dist
        eyes_center = tuple(eyes_center[:2])
        M = cv2.getRotationMatrix2D(eyes_center, roll, scale)
        tX = self.target_size[0] * 0.5
        tY = self.target_size[1] * self.target_eye_center
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        output = cv2.warpAffine(img, M, self.target_size,
                                flags=cv2.INTER_CUBIC)
        return output


def face_orientation(l_eye, r_eye, mouth):
    l_eye_center = np.mean(l_eye, axis=0)
    r_eye_center = np.mean(r_eye, axis=0)
    mouth_center = np.mean(mouth, axis=0)
    dY = r_eye_center[1] - l_eye_center[1]
    dX = r_eye_center[0] - l_eye_center[0]

    roll = 90 if dX == 0 else np.degrees(np.arctan(dY / dX))
    p1 = np.array(r_eye_center)
    p2 = np.array(l_eye_center)
    p3 = np.array(mouth_center)

    # These two vectors are in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    cp_norm = np.linalg.norm(cp)
    cp = cp / cp_norm
    yaw = np.degrees(np.arcsin(cp[0]))
    pitch = np.degrees(np.arcsin(cp[1]))

    print("roll: %d, pitch %d, yaw %d" % (roll, pitch, yaw))
    return roll, pitch, yaw


if __name__ == "__main__":
    visualizer = visualization.VisualizerOpencv()

    with open("data/preds.json") as f:
        data = json.load(f)
    fa = FaceAligner()
    for img_name, ff in data.items():
        image = cv2.imread(os.path.join("data/children", img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ff = np.array(ff)
        image, _ = visualizer.plt_img(image, [], landmarks=ff)
        print(image)
        img = fa.align_face(image, ff[43:48], ff[36:42], ff[8])

        image = cv2.resize(image, None, fx=3, fy=3)
        img = cv2.resize(img, None, fx=3, fy=3)

        cv2.imshow(img_name, img)
        cv2.imshow(img_name + "2", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # with open("data/preds.json") as f:
    #     data = json.load(f)
    # fa = FaceAligner()
    # for img_name, ff in data.items():
    #     print(img_name)
    #     image = cv2.imread(os.path.join("data/children", img_name))
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ff = np.array(ff)
    #
    #
    #     face_orientation(ff[43:48], ff[36:42], ff[8])
    #     img = visualizer.plt_img(image, [], landmarks=ff, title=img_name)

    cv2.waitKey(0)
    # cv2.imwrite(os.path.join("data/fa_out",img_name), img)
