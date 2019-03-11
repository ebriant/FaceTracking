import cv2
import numpy as np
import json
import os
import visualization


def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
    self.desired_left_eye = desired_left_eye
    self.desired_face_width = desired_face_width
    self.desired_face_height = desired_face_height

    if self.desired_face_height is None:
        self.desired_face_height = self.desired_face_width


def align(self, image, left_eye_center, right_eye_center):
    left_eye_center = np.mean(left_eye_center, axis=0)
    right_eye_center = np.mean(right_eye_center, axis=0)

    # compute the angle between the eye centroids
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - self.desired_left_eye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - self.desired_left_eye[0])
    desiredDist *= self.desired_face_width
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = self.desired_face_width * 0.5
    tY = self.desired_face_height * self.desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (self.desired_face_width, self.desired_face_height)
    output = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_CUBIC)
    # return the aligned face
    return output


class FaceAligner:
    def __init__(self, target_height=0.8, target_eye_center=0.6):
        self.target_height = target_height
        self.target_eye_center = target_eye_center

    def align_face(self, img, l_eye, r_eye, chin):
        r_eye_center = np.mean(l_eye, axis=0)
        l_eye_center = np.mean(r_eye, axis=0)
        dY = r_eye_center[1] - l_eye_center[1]
        dX = r_eye_center[0] - l_eye_center[0]

        roll = 90 if dX==0 else np.degrees(np.arctan(dY / dX))

        eyes_center = np.mean([l_eye_center, r_eye_center], axis=0)

        dist = np.linalg.norm((eyes_center-chin)[:2])
        dist = dist / img.shape[1]
        scale = self.target_height / dist

        M = cv2.getRotationMatrix2D(eyes_center, roll, scale)
        output = cv2.warpAffine(image, M, (img.shape[0], img.shape[1]),
                                flags=cv2.INTER_CUBIC)

        return output

def face_orientation(l_eye, r_eye, chin):
    l_eye_center = np.mean(l_eye, axis=0)
    r_eye_center = np.mean(r_eye, axis=0)
    dY = r_eye_center[1] - l_eye_center[1]
    dX = r_eye_center[0] - l_eye_center[0]

    if dX == 0:
        roll = 90
    else:
        roll = (np.degrees(np.arctan(dY / dX))) % 360

    p1 = np.array(r_eye_center)
    p2 = np.array(l_eye_center)
    p3 = np.array(chin)

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


if __name__ == "__main__":
    visualizer = visualization.VisualizerOpencv()

    with open("data/preds.json") as f:
        data = json.load(f)
    print(data)
    fa = FaceAligner()
    for img_name, ff in data.items():
        image = cv2.imread(os.path.join("data/children", img_name))
        img = fa.align_face(image, ff[43:48], ff[36:42], ff[8])
        visualizer.plt_img(image, [], landmarks=ff, title=img_name)

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
