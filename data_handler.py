import json
import config
import os


# Data should be x y w h
class DataManager:
    def __init__(self, dir):
        self.dir = dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def write_data(self, data, video, frame):
        file = "%s_%s.json" % (video, frame)
        path = os.path.join(self.dir, file)

        with open(path, 'w') as outfile:
            json.dump(data, outfile)
        print("file written")

    def get_data(self, file_path):
        if file_path[:-5] == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, "r") as f:
                data = eval(f.read())
        return data
