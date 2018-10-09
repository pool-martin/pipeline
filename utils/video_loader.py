import numpy as np
import skvideo.io
import cv2
from queue import Queue
from threading import Thread, Event
import os
import time

class VideoLoader:
    def __init__(self, path, queueSize=128, videos_to_load=None, frame_shape=None, stop_event=None):
        self.dataset = {}
        self.stopped = False
        self.videos_to_load = videos_to_load
        self.path = path
        self.frame_shape = frame_shape
        self.stop_event = stop_event

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
    def start(self):
        print('#################################### VideoLoader Start')
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, name='VideoLoader', args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        for video in self.videos_to_load:
            video_path = os.path.join(self.path, 'videos', '{}.mp4'.format(video))
            print('starting loading {}'.format(video_path))
            outputdict = {}
            if self.frame_shape:
                outputdict['-s'] = '{}x{}'.format(self.frame_shape, self.frame_shape)
            self.dataset[video] = skvideo.io.vread(video_path,  outputdict=outputdict)
            print('finished loading {}'.format(video_path))
            if self.stop_event.is_set():
                break
        self.stop()
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def get_video_frames(self, video, frames_identificator, snippet_path, image_size, split_type):

        video = video.decode("utf-8") 
        print('#################################### GET_VIDEO_FRAME {}'.format(video))

        # while not self.stopped:
        #     if video in self.dataset or self.stop_event.is_set():
        #         print('video {} loaded'.format(video))
        #         break
        #     else:
        #         print('not finished to load dataset')
        #         time.sleep(10)

        if(split_type.decode("utf-8") == '2D'):
            frame_numbers = [frames_identificator]
        elif(split_type.decode("utf-8") == '3D'):
            with open(snippet_path.decode("utf-8"), 'r') as f:
                frame_numbers = f.read().split('\n')[:-1]
            # print(video_path)
            # print( frame_numbers )
            frame_numbers = [float(number) for number in frame_numbers]


        return np.take(self.dataset[video], indices=frame_numbers, axis=0)