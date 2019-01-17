import numpy as np
import skvideo.io
import cv2
import optflow
from queue import Queue
from threading import Thread, Event, Lock
from subprocess import call
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
        self.load_lock = Lock()
        self.sum_lock = Lock()
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

    def get_video_frames(self, video_name, frames_identificator, snippet_path, image_size, split_type, fragments_count, debug_flag):

        video = video_name.decode("utf-8") 
        # print('#################################### GET_VIDEO_FRAME {} fragment {} last {}'.format(video, snippet_path.decode("utf-8"), fragments_count))
        if debug_flag: print('\nvideo {}'.format(video))

        # every try that cames here when video is not loaded will wait here
        if video not in self.dataset:
            # if debug_flag: print('Before the lock: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
            try:
                self.load_lock.acquire()
                # only the first will actually load the video.
                if video not in self.dataset:
                    if debug_flag: print('\nFirst loading: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
                    # with open('/DL/2kporn/dimensions_video/{}.etf'.format(video), 'r') as f:
                    #     dimensions = f.read().split('\n')[0]
                    # self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)), height=dimensions.split('x')[0], width=dimensions.split('x')[1], outputdict=self.outputdict) #, backend='ffmpeg', verbosity=1)
                    self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)),  outputdict=self.outputdict) #, backend='ffmpeg', verbosity=1)
                    if(self.dataset.get(video, None) is not None):
                      if debug_flag: print('\nvideo {} shape {}'.format(video, self.dataset[video].shape))
                    self.fragments[video] = 0
                    # time.sleep(1)
            finally:
                self.load_lock.release()

        # if debug_flag: print('After the lock: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))

        if(split_type.decode("utf-8") == '2D'):
            frame_numbers = [frames_identificator]
        elif(split_type.decode("utf-8") == '3D'):
            with open(snippet_path.decode("utf-8"), 'r') as f:
                frame_numbers = f.read().split('\n')[:-1]
            frame_numbers = [float(number) for number in frame_numbers]

        with self.sum_lock:
            if(self.dataset.get(video, None) is not None):
              if debug_flag: print('take video {} shape {} frames {}'.format(video, self.dataset[video].shape, frame_numbers), flush=True)
              fragment = np.take(self.dataset[video], indices=frame_numbers, axis=0)
              self.fragments[video] += 1
        # if debug_flag: print('After take the fragment : video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))

        # print('\n\n fragment raw \n{}'.format(fragment))
        if (self.fragments.get(video, None) is not None) and self.fragments[video] == fragments_count:
          if debug_flag: print('\nLAST FRAGMENT: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))
          self.dataset[video] = None
          self.fragments[video] = 0
          gc.collect()
          self.dataset.pop(video, None)
            # print('\n\n fragment float32 \n{}'.format(fragment.astype('float32')))

        return fragment.astype('float32')

    def get_video_flows(self, video_name, frames_identificator, snippet_path, image_size, split_type, fragments_count, debug_flag, of_difference):

        video = video_name.decode("utf-8") 
        fragment_dir = '/Exp/2kporn/cache/of/{}'.format(video)
        fragment_path = '{}/{}'.format(fragment_dir, frames_identificator)

        # Cache
        if (os.path.isfile(fragment_path)):
            with open(fragment_path, 'rb') as f:
                results = f.read()
            return results
        
        if not os.path.isdir(fragment_dir):
            command = "mkdir -p " + fragment_dir
            # print('\n', command)
            call(command, shell=True)

        # print('#################################### GET_VIDEO_FRAME {} fragment {} last {}'.format(video, snippet_path.decode("utf-8"), fragments_count))
        if debug_flag: print('\nvideo {}'.format(video))

        # every try that cames here when video is not loaded will wait here
        if video not in self.dataset:
            # if debug_flag: print('Before the lock: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
            try:
                self.load_lock.acquire()
                # only the first will actually load the video.
                if video not in self.dataset:
                    if debug_flag: print('\nFirst loading: video {} fragment {} count {}'.format(video, frames_identificator, fragments_count))
                    # with open('/DL/2kporn/dimensions_video/{}.etf'.format(video), 'r') as f:
                    #     dimensions = f.read().split('\n')[0]
                    # self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)), height=dimensions.split('x')[0], width=dimensions.split('x')[1], outputdict=self.outputdict) #, backend='ffmpeg', verbosity=1)
                    self.dataset[video] = skvideo.io.vread(os.path.join(self.path, 'videos', '{}.mp4'.format(video)),  outputdict=self.outputdict) #, backend='ffmpeg', verbosity=1)
                    if(self.dataset.get(video, None) is not None):
                      if debug_flag: print('\nvideo {} shape {}'.format(video, self.dataset[video].shape))
                    self.fragments[video] = 0
                    # time.sleep(1)
            finally:
                self.load_lock.release()

        # if debug_flag: print('After the lock: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))


        with open(snippet_path.decode("utf-8"), 'r') as f:
            frame_numbers = f.read().split('\n')[:-1]
        
        position = 0
        if of_difference == 1:
            position = 1
        elif of_difference == 5:
            position = 2
        elif of_difference == 15:
            position = 3
        elif of_difference == 30:
            position = 4

        frame_numbers = frame_numbers[0] # Just 2D now
        frames = frame_numbers.split('|')
        frame_1 = float(frames[0])
        frame_2 = float(frames[position])

        frame_numbers = [frame_1, frame_2]
        with self.sum_lock:
            if(self.dataset.get(video, None) is not None):
              if debug_flag: print('take video {} shape {} frames {}'.format(video, self.dataset[video].shape, frame_numbers), flush=True)
              fragment = np.take(self.dataset[video], indices=frame_numbers, axis=0)
              self.fragments[video] += 1
        # if debug_flag: print('After take the fragment : video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))

        # print('\n\n fragment raw \n{}'.format(fragment))
        if (self.fragments.get(video, None) is not None) and self.fragments[video] == fragments_count:
          if debug_flag: print('\nLAST FRAGMENT: video {} fragment {} count {} total {}'.format(video, frames_identificator, self.fragments[video], fragments_count))
          self.dataset[video] = None
          self.fragments[video] = 0
          gc.collect()
          self.dataset.pop(video, None)
            # print('\n\n fragment float32 \n{}'.format(fragment.astype('float32')))

        fragment_0 = cv2.cvtColor(fragment[0], cv2.COLOR_BGR2GRAY)
        fragment_1 = cv2.cvtColor(fragment[1], cv2.COLOR_BGR2GRAY)

        # flow = cv2.calcOpticalFlowFarneback(fragment_0,fragment_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = optflow.brox(video_frames[0] / 255., video_frames[1] / 255.)

        numpy_flow = np.asarray(flow, dtype=np.float32)
        final_fragment = np.stack([numpy_flow], axis=0)
        # print('numpy_flow shape', numpy_flow.shape)

        with open(fragment_path, 'wb') as f:
        f.write(final_fragment)

        return final_fragment