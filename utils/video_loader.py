import numpy as np
import skvideo.io
import cv2
from queue import Queue
from threading import Thread, Event, Lock
import os
import time
import gc

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

class VideoLoader:
    def __init__(self, path, frame_shape=None, stop_event=None):
        self.dataset = {}
        self.fragments = {}
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

    def get_video_frames(self, video, frames_identificator, snippet_path, image_size, split_type, fragments_count, debug_flag):

        load_lock = Lock()
        sum_lock = Lock()
        video = video.decode("utf-8") 
        # print('#################################### GET_VIDEO_FRAME {} fragment {} last {}'.format(video, snippet_path.decode("utf-8"), fragments_count))

        # everi try that cames here when video is not loaded will wait here
        if video not in self.dataset:
            if debug_flag: print('Before the lock: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
            try:
                load_lock.acquire()
                # only the first will actually load the video.
                if video not in self.dataset:
                    if debug_flag: print('The very first loading: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
                    self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)),  outputdict=self.outputdict)
                    self.fragments[video] = 0
                    time.sleep(1)
            finally:
                load_lock.release()

        if debug_flag: print('After the lock: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))

        if(split_type.decode("utf-8") == '2D'):
            frame_numbers = [frames_identificator]
        elif(split_type.decode("utf-8") == '3D'):
            with open(snippet_path.decode("utf-8"), 'r') as f:
                frame_numbers = f.read().split('\n')[:-1]
            frame_numbers = [float(number) for number in frame_numbers]

        with sum_lock:
            fragment = np.take(self.dataset[video], indices=frame_numbers, axis=0)
            self.fragments[video] += 1
        if debug_flag: print('After take the fragment : video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))


        if self.fragments[video] == fragments_count:
          if debug_flag: print('LAST FRAGMENT: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))
          self.dataset[video] = None
          self.fragments[video] = 0
          gc.collect()

        return fragment.astype('float32')