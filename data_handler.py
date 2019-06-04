import config
import os


# Data should be x y w h
class DataHandler:
    def __init__(self, path):

        self.dump_file = path
        with open(self.dump_file, "w+"):
            return

    def write_data(self, bbox):
        with open(self.dump_file, "a") as f:
            f.write("%d %d %d %d\n" % tuple(bbox))


def save_data(dump_file_path, data):
    with open(dump_file_path, "w+") as f:
        f.write(str(data))

def get_data(file):
    with open(file, "r") as f:
        data = eval(f.read())
    return data
