import config
import os


# Data should be x y w h
class DataHandler:
    def __init__(self, name):
        self.dump_file = os.path.join(config.tracking_data, "%s.txt" % name)

        with open(self.dump_file, "w+"):
            return

    def write_data(self, bbox):
        with open(self.dump_file, "a") as f:
            f.write("%d %d %d %d\n" % tuple(bbox))
