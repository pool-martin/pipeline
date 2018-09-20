import cv2, os
import numpy as np
import tensorflow as tf

def get_video_params(video_path):
    # print('.', end='', flush=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height =  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return frame_count, fps, height, width


def get_video_frames(video_path, frames_identificator, snippet_path, image_size, split_type):
    video_frames = []

    video_path = video_path.decode("utf-8") 

    if(split_type.decode("utf-8") == '2D'):
        frame_numbers = [frames_identificator]
    elif(split_type.decode("utf-8") == '3D'):
        with open(snippet_path.decode("utf-8"), 'r') as f:
            frame_numbers = f.read().split('\n')[:-1]
        # print(video_path)
        # print( frame_numbers )
        frame_numbers = [float(number) for number in frame_numbers]

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False): raise ValueError('Error opening video {}'.format(video_path))

    for frame_no in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if (ret == False): raise ValueError('Error extracting video {} frame {}'.format(video_path, frame_no))

        # unfortunately opencv uses bgr color format as default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, tuple(image_size), interpolation=cv2.INTER_CUBIC)

        # adhere to TS graph input structure
        numpy_frame = np.asarray(frame)
        #numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        #norm_image = cv2.normalize(numpy_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # numpy_frame = np.expand_dims(numpy_frame, axis=0)
        video_frames.append(numpy_frame.astype('float32'))

    cap.release()
    results = np.stack(video_frames, axis=0)
    # print(len(results), results.shape)
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

if __name__ == '__main__':
    show_video_frame('/home/jp/repos/DL/2kporn/videos/vNonPorn000001.mp4', 800)