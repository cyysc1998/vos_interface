import argparse
import cv2
import os
import json


def parse_args():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--input_file', type=str, default='./')
    args = arg_parse.parse_args()
    return args


def get_image_info(file_path):
    videos = {}
    video_list = [path for path in os.listdir(file_path) if not path.endswith('txt')]
    for video in video_list:
        img_path = os.path.join(file_path, video, 'color')
        clip_frame = len(os.listdir(img_path))
        img_list = os.listdir(img_path)
        img_size = None
        for img in img_list:
            image = os.path.join(img_path, img)
            img = cv2.imread(image)
            if img_size is None:
                img_size = img.shape
            else:
                if img.shape != img_size:
                    print('Error: image size is not consistent')
                    exit(1)
        videos[video] = {
            'clip_frame': clip_frame,
            'img_size': img_size[:2]
        }
    with open('video_info.json', 'w') as f:
        json.dump(videos, f, indent=4)

    
        

if __name__ == '__main__':
    args = parse_args()
    get_image_info(args.input_file)

