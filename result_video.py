import cv2
import numpy as np
import config
import os

dir1 = os.path.join(config.data_dir, "output_memtrack")
dir2 = os.path.join(config.data_dir, "output")

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/output.avi',fourcc, 20.0, (640,480))

dir1_images = os.listdir(dir1)
dir2_images = os.listdir(dir2)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)


        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


vis = np.concatenate((img1, img2), axis=1)
