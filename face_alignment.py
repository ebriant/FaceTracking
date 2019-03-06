import cv2
import numpy as np
import json
import os

class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width

    def align(self, image, left_eye_center, right_eye_center):

        # for i in left_eye_center:
        #     cv2.circle(image, (int(i[0]), int(i[1])), 2, color=(0, 0, 255))

        print(left_eye_center)
        left_eye_center = np.mean(left_eye_center, axis=0)
        print(left_eye_center)

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


if __name__ == "__main__":
    with open("data/preds.json") as f:
        data = json.load(f)

    print(data)
    fa = FaceAligner()
    for img_name, ff in data.items():
        image = cv2.imread(os.path.join("data/", img_name))
        img = fa.align(image, ff[43:48], ff[36:42])
        cv2.imwrite(os.path.join("data/fa_out",img_name), img)




