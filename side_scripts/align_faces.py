import faceAlignment.face_alignment as f_a
from face_alignment import FaceAligner
import cv2
import matplotlib.image as mpimg
import os

fa = f_a.FaceAlignment(f_a.LandmarksType._3D, device='cuda:0', flip_input=True)
face_aligner = FaceAligner(target_height=0.4, target_eye_center=0.35)

dir = "C:/Users/Horio/Documents/Children_faces"
out = "C:/Users/Horio/Documents/Children_faces/out"

print(os.listdir(dir))
for im in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, im)):
        img = cv2.imread(os.path.join(dir, im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = fa.get_landmarks(img)[-1]
        aligned_face_img = face_aligner.align_face(img, preds[43:48], preds[36:42], preds[8])
        aligned_face_img = cv2.cvtColor(aligned_face_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(out, im[:-4]+".jpg"), aligned_face_img)
