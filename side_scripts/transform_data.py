import argparse
import os
import config
import data_handler
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiwi Training')

    parser.add_argument('-f', '--file', type=str, help='')
    parser.add_argument('-s', '--start', default=0, type=int, help='')
    parser.add_argument('-o', '--out_dir', type=str, help='')
    parser.add_argument('-v', '--video', type=str, help='')
    args = parser.parse_args()

    data = data_handler.get_data(args.file)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    l = len(data[next(iter(data))][config.BBOX_KEY])
    for i in range(l):
        frame_data = {}
        for name, item in data.items():
            frame_data[name] = {config.BBOX_KEY: [item[config.BBOX_KEY][i]]}

        path = os.path.join(args.out_dir, "%s_%d.json" % (args.video, i))
        with open(path, 'w') as outfile:
            json.dump(frame_data, outfile)
        print("file written", path)
