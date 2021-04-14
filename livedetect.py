from aiohttp import web
import socketio
import cv2
import asyncio
from utils.VideoToImages import *
from utils.CustomVideoDataset import *
from utils.WarmupMultiStepLR import *
from utils.MetricLogger import *

import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torchvision
from torch import nn
from collections import deque
import time
import errno
import math
import shutil


"""
We will be using a socketIO server written in python and a javascript based socketIO client, to the versions must be compatible
python-socketio==4.6.0
javascript: https://cdnjs.cloudflare.com/ajax/libs/socket.io/2.2.0/socket.io.js
as per this https://python-socketio.readthedocs.io/en/latest/intro.html#what-is-socket-io
python-socketio 4.x should be used with javascript sockeIO 1.x and 2.x
"""
connection_flag = 0
#global variable, the model that is used to make predictions
prediction_model = None
fps = None
curr_frame_pos = None
frame_count = None

def collate_fn(batch):
    return default_collate(batch)

def remdir(path):
    try:
        if os.path.exists(path):
            print("Removing dir =", path)
            shutil.rmtree(path)
    except OSError as e:
        print(e.errno)
        raise

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

"""
    The dataset of the frames from the carmera used for prediction
"""
class FrameDataset():

    def __init__(self, max_frames, fps):
        self.framequeue = deque()
        self.max_frames = max_frames
        self.fps = fps

    def add_frame(self, frame):
        if self.num_frames() > self.max_frames:
            self.framequeue.popleft()
        self.framequeue.append(frame)

    def num_frames(self):
        return len(self.framequeue)

    def get_frames(self):
        return [self.framequeue[i] for i in range(self.num_frames())]

    def get_live_frames_as_images(self):
        annotationsfile = "annotations.txt"
        temp_path = "temp"
        remdir(temp_path)
        image_temp_dir = temp_path + "/" + "cat"
        makedir(image_temp_dir)
        frames = self.get_frames()
        annotations = []
        batch_idx = 0
        image_temp_path = image_temp_dir + "/" + str(batch_idx)
        makedir(image_temp_path)
        for idx in range(len(frames)):
            frame = frames[idx]
            im = Image.fromarray(frame)
            fullpath = image_temp_path + "/" + "img_{}.jpg".format(idx)
            im.save(fullpath)
        image_rel_path = "cat" + "/" + str(batch_idx)
        #setting the class to 0, we do not care about the class since we will use this for testing only
        an = annots(path=image_rel_path, start=0, end=len(frames), cls=0, fps=self.fps)
        annotations.append(an)
        annotationsfilepath = temp_path + "/" + annotationsfile
        with open(annotationsfilepath, 'a') as f:
            f.write('[')
            cntr = 0
            num_ele = len(annotations)
            for an in annotations:
                f.write(an.toJSON())
                if cntr < num_ele - 1:
                    f.write(',')
                cntr += 1
            f.write(']')
        return image_temp_path, annotationsfilepath

    def get_dataset_batches(self, min_frames=16):
        batches = []
        transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
        # all frames buffered until this point
        frames = self.get_frames()
        idxs = np.arange(len(frames))
        idxs = torch.Tensor(idxs)
        idxs = idxs.type(torch.LongTensor)
        # create clips with sequence of frames, each clip is 16 frames, mutiple clips spanning entire bufferred video
        clips_idxs = unfold(idxs, 16, 1)
        clips_idxs = clips_idxs.type(torch.uint8)
        frames_batches = []
        for clip_idx in clips_idxs:
            frames_batches.append([frames[i] for i in clip_idx.type(torch.uint8).numpy()])
        height, width, channels = frames[0].shape
        #need to parallelize this routine, this one takes about 1.2 seconds for 8 batches of 16 frames each
        batches = []
        for batch_idx in range(len(frames_batches)):
            imageseq = frames_batches[batch_idx]  # a seq of 16 images / frames in this batch
            framestensor = torch.FloatTensor(16, height, width, channels)
            start_time = time.time()
            for idx in range(len(imageseq)):
                frame = imageseq[idx]
                frame = torch.from_numpy(frame)
                framestensor[idx, :, :, :] = frame.type(torch.uint8)
            batches.append(transform_eval(framestensor.type(torch.uint8)))
            print("Took {} seconds to for idx in range(len(imageseq))".format(time.time() - start_time))
        #convert batches to tensor and ship
        c, b, h, w = batches[0].shape
        batchtensor = torch.FloatTensor(len(clips_idxs), c, b, h, w)
        for i in range(len(batches)):
            batchtensor[i, :, :, :, :] = batches[i]
        return batchtensor

    def get_as_dataset(self, min_frames=16):
        transform_eval = VideoClassificationPresetEval((128, 171), (112, 112))
        frames = self.get_frames()
        height, width, channels = frames[0].shape
        framestensor = torch.FloatTensor(min_frames, height, width, channels)
        # frame = cv2.imread(image_name)
        for idx in range(min_frames - 1):
            frame = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            framestensor[idx, :, :, :] = frame.type(torch.uint8)
        return transform_eval(framestensor.type(torch.uint8))


all_frames = None #at 15 fps, 3 seconds
#code from here https://python-socketio.readthedocs.io/en/latest/intro.html#what-is-socket-io
#and here https://tutorialedge.net/python/python-socket-io-tutorial/
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

cap = None
local_video_path = None

def gen_frames():
    global all_frames
    global curr_frame_pos
    while True:
        success, frame = cap.read()  # read a frame from the camera
        #print("Current frame pos = {}".format(curr_frame_pos))
        curr_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # ret, buffer = cv2.imencode('.jpg', frame)
        # buffer = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        all_frames.add_frame(frame)
        if not success:
            break
        else:
            frame = cv2.resize(frame, (480, 320))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# sio.on('feed')
# async def send_position():
#     await asyncio.sleep(0.06)
#     return curr_frame_pos

sio.on('feed')
async def video_feed(request):
    response = web.StreamResponse()
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    await response.prepare(request)

    for frame in gen_frames():
        await asyncio.sleep(0.06)
        await response.write(frame)
        if local_video_path is not None:
            msg = "\'{\"position\":" + str(curr_frame_pos) + ",\"frame_count\":" + str(frame_count) + "}'"
            #msg = '{"position":12.0,"frame_count":19225.0}'
            await sio.emit('feed', eval(msg))
    return response


async def index(request):
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('connect')
async def connect_handler(sid, environ):
    global connection_flag
    print("New connection from {}".format(sid))
    connection_flag = 1
    await sio.emit('welcome', "connected")

@sio.event
def disconnect(sid):
    global connection_flag
    print('disconnect ', sid)
    connection_flag = 0

@sio.on('skip')
async def print_message(sid, message):
    global cap
    global curr_frame_pos
    if local_video_path != None:
        print("Socket ID: " , sid)
        ## await a successful emit of our reversed message
        ## back to the client
        print("Skipping 10 frames...")
        curr_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_pos + 75)
    #await sio.emit('message', message[::-1])

async def send_message():
    global connection_flag
    global all_frames
    global curr_frame_pos
    try:

        print("Background task started...")
        await asyncio.sleep(1)
        print("Wait till a client connects...")
        while connection_flag == 0:
            await asyncio.sleep(0.1)
            pass
        while True:
            start_time_routine = time.time()
            criterion = nn.CrossEntropyLoss()
            if all_frames.num_frames() >= 16:
                start_time_subroutine = time.time()
                start_time = time.time()
                batches = all_frames.get_dataset_batches()
                print("Took {} seconds to get_dataset_batches()".format(time.time() - start_time))
                with torch.no_grad():
                    m = prediction_model.get_model()
                    m.eval()
                    #move to GPU
                    start_time = time.time()
                    print("Shape of video = {}".format(batches.shape))
                    batches = batches.to(prediction_model.device, non_blocking=True)
                    target = torch.Tensor(np.ones(len(batches)))
                    target = target.type(torch.long)
                    target = target.to(prediction_model.device, non_blocking=True)
                    output = m(batches)
                    time_diff = time.time() - start_time
                    print("predicted an input of shape {} in {} seconds".format(batches.shape, time_diff))
                    start_time = time.time()
                    loss = criterion(output, target)
                    _, pred = output.topk(5, 1, True, True)
                    print("Took {} seconds to predictions".format(time.time() - start_time))
                    #print("pred = {}".format(pred))

                # t = torch.Tensor(np.ones(len(batches)))
                # t = t.type(torch.long)
                # t = t.to(prediction_model.device, non_blocking=True)
                # loss = criterion(output, t)
                # _, pred = output.topk(5, 1, True, True)
                # p = torch.nn.functional.softmax(output, dim=1)
                # vals, preds = p.topk(5, 1, True, True)
                # msg = "Best class = {}, best prob = {}, frame_idx={}".format(preds[0][0], vals[0][0], curr_frame_pos)
                msg = "Best class = {}, best prob = {}, frame_idx={}".format(1, 0.01, 10)
                await sio.emit('feedback', msg)
                print("Took {} seconds to subroutine send_message()".format(time.time() - start_time_subroutine))
            await asyncio.sleep(0.6)
            print("Took {} seconds to send_message()".format(time.time() - start_time_routine))

    finally:
        print("Background task exiting!")

# @sio.on('message')
# async def print_message(sid, message):
#     print("Socket ID: {}".format(sid))
#     print(message)
#     await sio.emit('message', message[::-1])

app.router.add_get('/', index)
app.router.add_get('/videostream', video_feed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Video classification training using image sequences')

    parser.add_argument('--workers', default=4, help='number of workers')
    parser.add_argument('--model', default='r2plus1d_18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--clip-len', default=16, type=int, metavar='N',
                        help='number of frames per clip')
    parser.add_argument('--clips-per-video', default=5, type=int, metavar='N',
                        help='maximum number of clips per video to consider')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume-dir', default='checkpoint.pth', help='path where the model checkpoint is saved')
    parser.add_argument('--saved-video', default='', help='a saved video to use for streaming inference')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--fromlocalvideo",
        dest="fromlocalvideo",
        help="Use a saved video instead of the live camera",
        action="store_true",
    )

    args = parser.parse_args()

    return args

class Model():
    def __init__(self, args):
        self.model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
        self.device = torch.device(args.device)
        self.model.to(self.device)
        if args.resume_dir:
            if not os.path.exists(args.resume_dir):
                raise OSError("Checkpoint file does not exist!")
            else:
                checkpoint_file = args.resume_dir
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                self.model.load_state_dict(checkpoint['model'])
    def get_model(self):
        return self.model





if __name__ == '__main__':
    #global prediction_model
    args = parse_args()
    if not args.fromlocalvideo:
        cap = cv2.VideoCapture(0)
    else:
        if os.path.exists(args.saved_video):
            local_video_path = args.saved_video
            cap = cv2.VideoCapture(args.saved_video)
        else:
            raise OSError("No saved video at {}".format(args.saved_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS = {}".format(fps))
    print("Loading the model for prediction...")
    prediction_model = Model(args)
    all_frames = FrameDataset(max_frames=22, fps = math.floor(fps))
    # device = torch.device(args.device)
    # # load the pretrained weights for the known model
    # model = torchvision.models.video.__dict__[args.model](pretrained=args.pretrained)
    # model.to(device)

    # if args.resume_dir:
    #     if not os.path.exists(args.resume_dir):
    #         raise OSError("Checkpoint file does not exist!")
    #     else:
    #         checkpoint_file = args.resume_dir
    #         checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'])
    # print("Model loaded from checkpoint!")

    sio.start_background_task(target=lambda: send_message())
    web.run_app(app)
    if (not args.fromlocalvideo) & (cap == None):
        cap.release()