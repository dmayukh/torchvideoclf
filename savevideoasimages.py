import os
import cv2
from PIL import Image
import errno
import math

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
def main(args):
    if not os.path.exists(args.video_path):
        raise OSError("--video-path does no exist!")
    if not args.out_dir:
        raise OSError("--out-dir is mandatory!")

    video_path = args.video_path
    outdir = args.out_dir
    makedir(outdir)

    cap = cv2.VideoCapture(video_path)
    cntr = 0
    clip_length = 45 #frames
    idx = 0
    savedir = outdir + "/" + str(idx)
    makedir(savedir)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cntr % clip_length == 0:
            #create dir
            didx = math.floor(cntr / clip_length)
            savedir = outdir + "/" + str(didx)
            makedir(savedir)
            idx = 0
        fullpath = savedir + "/" + "img_{}.jpg".format(idx)
        cntr += 1
        idx += 1
        im = Image.fromarray(frame)
        im.save(fullpath)
        print("Saved {}".format(fullpath))
        #cv2.imshow('frame', gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Save a local video as images')
    parser.add_argument('--video-path',  help='the video to convert')
    parser.add_argument('--out-dir', help='the path where to save the video')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)