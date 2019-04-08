import children_tracker

if __name__ == '__main__':
    main_tracker = children_tracker.MainTracker()
    main_tracker.tmp_track = {'a': {'bbox': [407, 205, 60, 47]}, 'b': {'bbox': [259, 124, 46, 45]}, 'c': {'bbox': [185, 220, 79, 61]},
     'd': {'bbox': [203, 503, 166, 99]}, 'e': {'bbox': [400, 585, 42, 55]}, 'f': {'bbox': [415, 311, 76, 49]}}

    fd_bbox_list = [[220, 540, 142, 93], [301, 478, 55, 41]]

    fd_bbox_list, corrected = main_tracker.correct_faces_by_iou(fd_bbox_list)

    print(fd_bbox_list)
    print(corrected)

    print(main_tracker.tmp_track)
