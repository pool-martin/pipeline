import numpy as np
import skvideo.io
import cv2
from queue import Queue
from threading import Thread, Event
import os
import time
import gc

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class VideoLoader:
    def __init__(self, path, frame_shape=None, stop_event=None):
        self.dataset = {}
        self.stopped = False
        self.path = path
        self.frame_shape = frame_shape
        self.stop_event = stop_event
        if self.frame_shape:
            outputdict = {}
            outputdict['-s'] = '{}x{}'.format(self.frame_shape, self.frame_shape)
            self.outputdict = outputdict
        else:
            self.outputdict = None

    # def start(self):
    #     print('#################################### VideoLoader Start')
    #     # start a thread to read frames from the file video stream
    #     t = Thread(target=self.update, name='VideoLoader', args=())
    #     t.daemon = True
    #     t.start()
    #     return self
    # def update(self):
    #     for video in self.videos_to_load:
    #         video_path = os.path.join(self.path, 'videos', '{}.mp4'.format(video))
    #         print('starting loading {}'.format(video_path))
    #         if video not in self.dataset:
    #             self.dataset[video] = skvideo.io.vread(video_path,  outputdict=self.outputdict)
    #         print('finished loading {}'.format(video_path))
    #         if self.stop_event.is_set():
    #             break
    #     self.stop()
    # def stop(self):
    #     # indicate that the thread should be stopped
    #     self.stopped = True

    def get_video_frames(self, video, frames_identificator, snippet_path, image_size, split_type, last_fragment):

        video = video.decode("utf-8") 
        print('#################################### GET_VIDEO_FRAME {} fragment {} last {}'.format(video, snippet_path.decode("utf-8"), last_fragment.decode("utf-8")))

        if video not in self.dataset:
            self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)),  outputdict=self.outputdict)

        if(split_type.decode("utf-8") == '2D'):
            frame_numbers = [frames_identificator]
        elif(split_type.decode("utf-8") == '3D'):
            with open(snippet_path.decode("utf-8"), 'r') as f:
                frame_numbers = f.read().split('\n')[:-1]
            # print(video_path)
            # print( frame_numbers )
            frame_numbers = [float(number) for number in frame_numbers]

        fragment = np.take(self.dataset[video], indices=frame_numbers, axis=0)

        if last_fragment.decode("utf-8") in snippet_path.decode("utf-8"):
          print('\nLAST FRAGMENT {} last {}'.format(video, last_fragment.decode("utf-8")))
          self.dataset[video].video = None
          gc.collect()

        return fragment