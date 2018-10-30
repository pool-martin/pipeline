import numpy as np
import skvideo.io
import cv2
from queue import Queue
from threading import Thread, Event
import os
import time

class FileVideoStream:
    def __init__(self, path, queueSize=128, frames_to_extract=None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path, path.shape[0])
        self.stopped = False
        self.frames_to_extract = frames_to_extract

        # initialize the queue used to store frames read from
        # the video file
        maxsize = queueSize if frames_to_extract == None else len(frames_to_extract)
        self.Q = Queue(maxsize=maxsize)
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        # keep looping infinitely
        for frame_no in self.frames_to_extract:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # read the next frame from the file
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            # if not grabbed:
            #     self.stop()
            #     return

            # add the frame to the queue
            self.Q.put([grabbed, frame])
        self.stop()
        return


    def isOpened(self):
        # check if the video was opened with success
        return self.stream.isOpened()
    def read(self):
        # return next frame in the queue
        return self.Q.get()
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class VideoLoader:
    def __init__(self, path, queueSize=128, videos_to_load=None, frame_shape=None, stop_event=None):
        self.dataset = {}
        self.stopped = False
        self.videos_to_load = videos_to_load
        self.path = path
        self.frame_shape = frame_shape
        self.stop_event = stop_event
        if self.frame_shape:
            outputdict = {}
            outputdict['-s'] = '{}x{}'.format(self.frame_shape, self.frame_shape)
            self.outputdict = outputdict
        else:
            self.outputdict = None

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
    def start(self):
        print('#################################### VideoLoader Start', flush=True)
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, name='VideoLoader', args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        print('videos_to_load {}'.format(self.videos_to_load))
        for video in self.videos_to_load:
            video_path = os.path.join(self.path, 'videos', '{}.mp4'.format(video))
            print('starting loading {}'.format(video_path), flush=True)
            if video not in self.dataset:
#                self.dataset[video] = skvideo.io.vread(video_path,  outputdict=self.outputdict)
              self.dataset[video] = np.fromfile(video_path, dtype=np.uint8, count=-1)
              print('Video {} loaded'.format(video), flush=True)
            if self.stop_event.is_set():
                break
        print('finished loading videos', flush=True)
        self.stop()
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def get_video_frames(self, video, frames_identificator, snippet_path, image_size, split_type):
        video_frames = []
        t1= time.time()

        video = video.decode("utf-8") 
        print('#################################### GET_VIDEO_FRAME {}'.format(video), flush=True)

        if video not in self.dataset:
            # self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)),  outputdict=self.outputdict)
            self.dataset[video] = np.fromfile(os.path.join(self.path, 'videos', '{}.mp4'.format(video)), dtype=np.uint8, count=-1)

        if(split_type.decode("utf-8") == '2D'):
            frame_numbers = [frames_identificator]
        elif(split_type.decode("utf-8") == '3D'):
            with open(snippet_path.decode("utf-8"), 'r') as f:
                frame_numbers = f.read().split('\n')[:-1]
            frame_numbers = [float(number) for number in frame_numbers]

        # return np.take(self.dataset[video], indices=frame_numbers, axis=0)
        print('frames to load {} {}'.format(video, frame_numbers))

        fvs = FileVideoStream(self.dataset[video], frames_to_extract=frame_numbers).start()

        if (fvs.isOpened() == False): raise ValueError('Error opening video {}'.format(video))

        for frame_no in frame_numbers:

            ret, frame = fvs.read()
            if (ret == False): 
                fvs.stop()
                print('Error extracting video {} id {} frame {}'.format(video, frames_identificator, frame_no))
                break

            # unfortunately opencv uses bgr color format as default
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_CUBIC)

            # adhere to TS graph input structure
            numpy_frame = np.asarray(frame)

            # numpy_frame = np.expand_dims(numpy_frame, axis=0)
            video_frames.append(numpy_frame.astype('float32'))

        # cap.release()
        fvs.stop()
        #temporary recovery to not break the pipeline
        if(len(video_frames) > 0 and len(video_frames) < len(frame_numbers)):
            while len(video_frames) < len(frame_numbers):
                video_frames.append(video_frames[0])

        results = np.stack(video_frames, axis=0)
        t2= time.time()
        print('---------{}-{}'.format(video, t2 - t1), flush=True)
        return results
