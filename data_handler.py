import config

# Data should be Img_name x y w h

class DataHandler():

    def __init__(self):
        self.dump_file = config.tracking_data

        with open(self.dump_file, "w+") as f:
            return
