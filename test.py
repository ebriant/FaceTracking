import cv2
import utils
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import visualization
import numpy as np
import PIL


from matplotlib import pyplot as plt

img = mpimg.imread("data/img/00000.jpg")
img = np.array(img)
# fig, ax = plt.subplots()
# ax.plot(np.random.rand(10))
# ax.imshow(img)
# plt.show()

bboxes_list = [[94, 296, 77, 98], [317, 472, 48, 63]]
img2, bbox = visualization.plt_img(img, bboxes_list, callback=True)

# print(bbox)
# visualization.plt_img(img, bboxes_list)
# #
bboxes_list = [[94, 296, 77, 98]]
visualization.plt_img(img, bboxes_list)

# img = cv2.imread("data/img/00001.jpg")
# bbox = [290, 95, 384, 167]
# croped = img[bbox[0]-30:bbox[2]+30, bbox[1]-30:bbox[3]+30]
#
#
# face, _ = utils.rotate_face(croped, bbox, img.shape[0])
#
# print(face.shape)
# cv2.imshow("abbb", croped)
# cv2.imshow("aaaa", face)
# cv2.waitKey()
# cv2.destroyAllWindows()

data = {"a":[{"bbox":[1,1,1,1]}, {"bbox":[2,2,2,2]}, {"bbox":None}]}