# import numpy as np
import skvideo.io
# import cv2
import os
import subprocess
import csv
from collections import namedtuple
Clip = namedtuple('Clip', 'video_name label start_frame end_frame width height left_distance top_distance dataset scene_number crop_number')

INPUT_FOLDER = "./VOI/"
OUTPUT_FOLDER = "./Crowd-11/"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def read_line(line):
    # print(line)
    video_name=line[0]
    label=line[1]
    start_frame=int(line[2])
    end_frame=int(line[3])
    left_distance=float(line[4])
    top_distance=float(line[5])
    width=float(line[6])
    height=float(line[7])
    dataset=line[8]
    scene_number=line[9]
    crop_number=str(int(line[10]))
    clip=Clip(video_name, label, start_frame, end_frame, width, height, left_distance, top_distance, dataset, scene_number, crop_number)
    return clip




def create_clips(clip):
    input_video_folder = INPUT_FOLDER + clip.dataset+'/'
    if not os.path.exists(input_video_folder + clip.video_name):
        return
    print(input_video_folder + clip.video_name)
    metadata = skvideo.io.ffprobe(input_video_folder+clip.video_name)


    if 'video' not in metadata.keys():
        return


    real_width = int(metadata['video']['@width'])
    real_height = int(metadata['video']['@height'])

    crop_width = clip.width * real_width
    crop_height = clip.width * real_height
    crop_left_distance = clip.left_distance * real_width
    crop_top_distance = clip.top_distance * real_height

    video = OUTPUT_FOLDER + clip.label + '_' + clip.scene_number + '_' + clip.crop_number + '_' + clip.video_name

    if os.path.exists(video):
        return


    if crop_height<224:
        print("...............too small: rescaling")
        subprocess.call('avconv -i %s -an -filter_complex "select=between(n\,%s\,%s),setpts=PTS-STARTPTS, crop=%s:%s:%s:%s, scale=-2:224" %s'%(
            input_video_folder+clip.video_name,
            clip.start_frame,
            clip.end_frame,
            crop_width,
            crop_height,
            crop_left_distance,
            crop_top_distance,
            video),
                        shell=True)
    else:
        subprocess.call('avconv -i %s -an -filter_complex "select=between(n\,%s\,%s),setpts=PTS-STARTPTS, crop=%s:%s:%s:%s" %s'%(
            input_video_folder+clip.video_name,
            clip.start_frame,
            clip.end_frame,
            crop_width,
            crop_height,
            crop_left_distance,
            crop_top_distance,
            video),
                        shell=True)



def main():
    file = 'preprocessing.csv'

    list_clips=[]
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for line in reader:
            clip=read_line(line)
            list_clips.append(clip)

    for clip in list_clips:
        create_clips(clip)

    
if __name__ == "__main__":
    main()