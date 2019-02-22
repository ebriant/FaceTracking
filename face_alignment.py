import cv2
import numpy as np
import json


class FaceAligner:
    def __init__(self, desired_left_eye=(0.4, 0.4), desired_face_width=256, desired_face_height=None):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width

    def align(self, image, left_eye_center, right_eye_center):


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
        print(type(output))
        print(output)
        return output

img_name = "a_001.jpg"

with open("data/preds.json") as f:
    data = json.load(f)

preds=data[img_name]

fa = FaceAligner()
img = fa.align(cv2.imread("data/a_001.jpg"), preds[42], preds[39])
print("aaaaaaa")
cv2.imshow("iii", img)
cv2.waitKey()
cv2.destroyAllWindows()


