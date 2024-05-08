"""
Cite:
    - Dense Optical Flow: Brox, T., Bruhn, A., Papenberg, N., & Weickert, J. (2004). High accuracy optical flow estimation based on a theory for warping.
        In European conference on computer vision (pp. 25-36). Springer, Berlin, Heidelberg.
    - https://github.com/MounirB/py-denseflow/blob/master/denseflow.py

Extracts the optical flow from the RGB videos using the Dual TV-L1 optical flow algorithm.

This script extracts the optical flow from the RGB videos using the Dual TV-L1 optical flow algorithm. The extracted optical flow is saved as a video file.

Example:
    python denseflow.py --dataset=Crowd-11 --data_root=/home/anish_a24/IISc/Project/crowd-action-and-behaviour-recognition/ --new_dir=flow
"""

import os
import numpy as np
import cv2
from multiprocessing import Pool
import argparse
import skvideo.io
import time

def ToImg(raw_flow, bound):
    """
    This function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    """

    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= -bound
    flow *= (255/float(2*bound))
    return flow

def flows_to_video(flows, image, save_dir, video_name, bound):
    """
    To save the optical flow images as a video
    :param flows: list of optical flow arrays
    :param image: raw image
    :param save_dir: directory to save the video
    :param video_name: name of the video
    :param bound: bi-bound parameter
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flow_video_name = os.path.join(save_dir, video_name + "_flow.mp4")
    height, width, _ = image.shape
    flow_video = cv2.VideoWriter(flow_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for flow in flows:
        flow_x = ToImg(flow[..., 0], bound)
        full_flow_frame = cv2.cvtColor(flow_x, cv2.COLOR_GRAY2BGR)
        flow_frame = full_flow_frame.astype(np.uint8)
        flow_video.write(flow_frame)

    flow_video.release()

def read_video(video_path):
    """
    Reads the video using cv2.VideoCapture().
    Args:
    - video_path: Path to the video file.
    Returns:
    - videocapture: List of frames extracted from the video.
    """
    videocapture = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        videocapture.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    cap.release()
    return videocapture

def dense_flow(augs):
    """
    To extract dense_flow images
    :param augs: the detailed arguments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        bound: bi-bound parameter
        flow_quantity: maximum number of flow frames to extract from the video clip
    :return: no returns
    """

    video_path, save_dir, bound, flow_quantity = augs
    video_name = os.path.basename(video_path)
    flowDTVL1_list = list()
    step = 0

    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    try:
        print("--------", video_path)
        videocapture = read_video(video_path)
    except Exception as e:
        print(e)
        print('{} read error! '.format(video_name))
        return 0
    # if extract nothing, exit!
    if sum(frame.sum() for frame in videocapture) == 0:
        print('Could not initialize capturing', video_name)
        exit()

    len_frame = len(videocapture)

    """
    step: num of frames between each couple of contiguous extracted frames. 
    Example if step = 10 : frame_0, frame_1, ...ignored frames..., frame_10, frame_11, ...ignored frames..., etc.
    """
    try:
        if flow_quantity == 0:
            step = 1
        elif flow_quantity > 1 and flow_quantity <= len_frame:
            step = len_frame//flow_quantity
    except:
        print('{} flow quanity is erroneous '.format(flow_quantity))
        return 0

    frame_num = 0
    image, prev_image, gray, prev_gray = None, None, None, None
    num0 = 0
    while True:
        if num0+1 >= len_frame:
            break

        prev_image = videocapture[num0]
        prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        image = videocapture[num0+1]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frame_0 = prev_gray
        frame_1 = gray
        dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flowDTVL1 = dtvl1.calc(frame_0,frame_1,None)
        flowDTVL1_list.append(flowDTVL1)
        frame_num += 1
        num0 += step

    print(save_dir)
    flows_to_video(flowDTVL1_list, image, save_dir, video_name.split('.')[0], bound)



def parse_args():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset', default='Crowd-11', type=str, help='set the dataset name, to find the data path')
    parser.add_argument('--data_root', default='Data', type=str)
    parser.add_argument('--new_dir', default='flow', type=str)
    parser.add_argument('--num_workers', default=1, type=int, help='num of workers to act multi-process')
    parser.add_argument('--step', default=1, type=int, help='gap frames. Useful for dense_flow')
    parser.add_argument('--bound', default=15, type=int, help='set the maximum of optical flow')
    parser.add_argument('--mode', default='run', type=str, help='set \'run\' if debug done, otherwise, set debug')
    parser.add_argument('--flow_quantity', default=0, type=int, help='Number of flow frames to extract from the RGB video. Useful for dense_flow_bis')
    args = parser.parse_args()

    return args

def get_video_list(videos_root):
    """
    :param videos_root: path to the rgb videos of a dataset (or the folder containing the subfolders of videos)

    Tip - Possible alternative with the use of os.walk :
        result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.mp4'))]
    """

    video_list = list()
    # for class_name in os.listdir(videos_root):
    #     class_path = os.path.join(videos_root, class_name)
    #     for video_ in os.listdir(class_path):
    #         video_list.append(os.path.join(class_path, video_))
    for video in os.listdir(videos_root):
        video_list.append(os.path.join(videos_root, video))
    video_list.sort()

    return video_list


if __name__ == '__main__' :
    # Count the elapsed time from the start to the end of the processing
    start_time = time.time()

    # Specify the arguments
    args = parse_args()
    data_root = os.path.join(args.data_root, args.dataset)
    videos_root = os.path.join(data_root, 'rgb')
    num_workers = args.num_workers
    step = args.step
    bound = args.bound
    new_dir = args.new_dir
    mode = args.mode
    flow_quantity = args.flow_quantity

    # Get the videos list
    video_list = get_video_list(videos_root)
    save_dirs = [os.path.dirname(video.split('.')[0]).replace("rgb", new_dir) for video in video_list]

    # Run the dense_flow algorithm in multiprocessing mode
    pool = Pool(num_workers)
    if mode == 'run':
        pool.map(dense_flow, zip(video_list, save_dirs, [bound]*len(video_list), [flow_quantity]*len(video_list)))
    else:
        # Debug (mode == 'debug')
        dense_flow((video_list[0], save_dirs[0], bound, flow_quantity))

    elapsed_time = time.time() - start_time
    print("The elapsed time is : " + str(elapsed_time))