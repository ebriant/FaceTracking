import cv2
import utils

img = cv2.imread("data/img/00001.jpg")
bbox = [290, 95, 384, 167]
croped = img[bbox[0]-30:bbox[2]+30, bbox[1]-30:bbox[3]+30]


face, _ = utils.rotate_face(croped, bbox, img.shape[0])

print(face.shape)
cv2.imshow("abbb", croped)
cv2.imshow("aaaa", face)
cv2.waitKey()
cv2.destroyAllWindows()
