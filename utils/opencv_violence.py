import cv2, os, time, sys
from threading import Thread
import numpy as np
import tensorflow as tf
from queue import Queue

def get_video_params(video_path):
    # print('.', end='', flush=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frame_count, fps, height, width

class FileVideoStream:
    def __init__(self, path, queueSize=125, frames_to_extract=None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        maxsize = queueSize
        self.Q = Queue(maxsize=maxsize)
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        # keep looping infinitely
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(0, 125):
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # read the next frame from the file
            (grabbed, frame) = self.stream.read()

            # if the `grabbed` boolean is `False`, then we have
            # reached the end of the video file
            if not grabbed:
                self.stop()
                return

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

def get_video_frames(video_path, image_size, split_type):
    video_frames = []

    t1= time.time()
    video_path = video_path.decode("utf-8") 

    print(video_path)
    
    fvs = FileVideoStream(video_path).start()

#    cap = cv2.VideoCapture(video_path)
    if (fvs.isOpened() == False): raise ValueError('Error opening video {}'.format(video_path))

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while fvs.more():

        ret, frame = fvs.read()
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        # ret, frame = cap.read()
        if (ret == False): 
            fvs.stop()
            print('Error extracting video {} frame {}'.format(video_path, i))
            break

        # unfortunately opencv uses bgr color format as default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_CUBIC)

        # adhere to TS graph input structure
        numpy_frame = np.asarray(frame)
        #numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        #norm_image = cv2.normalize(numpy_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # numpy_frame = np.expand_dims(numpy_frame, axis=0)
        video_frames.append(numpy_frame.astype('float32'))

    # cap.release()
    fvs.stop()
    #temporary recovery to not break the pipeline
    if(len(video_frames) > 0 and len(video_frames) < 125):
        while len(video_frames) < 125:
            video_frames.append(video_frames[0])

    results = np.stack(video_frames, axis=0)
    t2= time.time()
    print('---------{}-{}'.format(video_path.split('/')[-1], t2 - t1))
    return results


def show_video_frame(video_name, frame_no):
    #Open the video file
    cap = cv2.VideoCapture(str(video_name))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  int(cap.get(cv2.CAP_PROP_FPS))
    print('count: {}, fps: {}'.format(frame_count, fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
    ret, frame = cap.read()
    print('ret:', ret)

    #Set grayscale colorspace for the frame. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Cut the video extension to have the name of the video
    my_video_name = video_name.split(".")[0]

    #Display the resulting frame
    cv2.imshow(my_video_name+' frame '+ str(frame_no),gray)

    #Set waitKey 
    cv2.waitKey()

    #Store this frame to an image
#    cv2.imwrite(my_video_name+'_frame_'+str(frame_seq)+'.jpg',gray)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



def video_to_numpy_array(video_path, image_size):
    video_frames = []

    t1= time.time()
    video_path = video_path 

   
    fvs = FileVideoStream(video_path).start()

#    cap = cv2.VideoCapture(video_path)
    if (fvs.isOpened() == False): raise ValueError('Error opening video {}'.format(video_path))

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret = True
    while ret == True:
        ret, frame = fvs.read()
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        # ret, frame = cap.read()
        if (ret == False): 
            fvs.stop()
            print('Error extracting video {}'.format(video_path))
            break

        # unfortunately opencv uses bgr color format as default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        frame = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_CUBIC)

        # adhere to TS graph input structure
        numpy_frame = np.asarray(frame)
        #numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        #norm_image = cv2.normalize(numpy_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # numpy_frame = np.expand_dims(numpy_frame, axis=0)
        video_frames.append(numpy_frame.astype('float32'))

    # cap.release()
    fvs.stop()

    results = np.stack(video_frames, axis=0)
    np.save('./teste.np', results)
    t2= time.time()
    print('---------{}-{}'.format(video_path.split('/')[-1], t2 - t1))
    return results


if __name__ == '__main__':
    #show_video_frame('/home/jp/repos/DL/2kporn/videos/vNonPorn000001.mp4', 800)
    video_to_numpy_array('/DL/2kporn/videos/vNonPorn000001.mp4', 299)