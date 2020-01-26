import cv2
import config


class Video:
    def __init__(self, start=0, end=None):
        self.cap = cv2.VideoCapture(config.video_path)
        self.start = start
        self.end = end
        self._index = 0

    def __iter__(self):
        self._index = self.start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        return self

    def __getitem__(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.cap.read()
        if ret or (self.end is not None and self._index >= self.end):
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise IndexError()

    def __next__(self):
        ret, frame = self.cap.read()
        if ret or (self.end is not None and self._index >= self.end):
            self._index += 1
            return frame
        else:
            raise StopIteration

if __name__ == '__main__':
    for i, v in enumerate(Video(start=20)):

        print(i)
        if i > 50:
            break
