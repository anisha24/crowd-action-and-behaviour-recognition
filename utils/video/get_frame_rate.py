"""
Video frame rate extraction.

This script extracts the frame rate of a video file using OpenCV.

Example:
    python get_frame_rate.py --video_path=/home/anish_a24/IISc/Project/crowd-action-and-behaviour-recognition/Crowd-11/rgb/0_21_4_000455714.mp4
"""

import cv2
import argparse

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the frame rate of a video.")
    parser.add_argument("--video_path", help="Path to the video file.")
    args = parser.parse_args()
    video_path = args.video_path
    fps = get_frame_rate(video_path)
    if fps is not None:
        print("Frame rate of the video:", fps)