import cv2


def draw_bbox(img, bbox, label="", color=(0, 255, 0), thickness=2):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0] + 15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return


def plt_img(img, bboxes, classes=[], scores=[], title="image"):
    height = img.shape[0]
    width = img.shape[1]
    selected_bbox = []
    bboxes_px = []
    b, g, r = cv2.split(img)  # get b,g,r
    img = cv2.merge([r, g, b])  # switch it to rgb
    for i in range(bboxes.shape[0]):
        xmin = int(bboxes[i, 0] * height)
        ymin = int(bboxes[i, 1] * width)
        xmax = int(bboxes[i, 2] * height)
        ymax = int(bboxes[i, 3] * width)

        bbox = [xmin, ymin, xmax, ymax]
        bboxes_px.append(bbox)
        draw_bbox(img, bbox)

    def mouse_position(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for bbox in bboxes_px:
                if is_in_bbox(bbox, x, y):
                    draw_bbox(img, bbox, "selected", (255, 0, 0))
                    cv2.imshow(title, img)
                    selected_bbox.append(bbox)
                    print("++++++++++++  ", selected_bbox)

    def is_in_bbox(box, x, y):
        if box[0] <= y <= box[2] and box[1] <= x <= box[3]:
            return True
        return False

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_position)
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return selected_bbox
