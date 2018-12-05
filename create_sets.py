#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  create_splits.py
#  
#  Copyright 2018 Joao Paulo Martin <joao.paulo.pmartin@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; 
#  either version 2 of the License, or (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
#  PURPOSE.  See the GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth 
#  Floor, Boston, MA 02110-1301, USA.
#  
#  

''' 
Create Splits to be used in 2D or 3D CNN models. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com)
'''

import argparse, os, time, random, math
from subprocess import call
import decimal
from utils import opencv

def load_args():
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D or 3D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/2kporn/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/Exp/2kporn/splits/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1')
    ap.add_argument('-sr', '--sample-rate',
                                    dest='sample_rate',
                                    help='sample rate to be used in video frame sampling.',
                                    type=int, required=False, default=1)
    ap.add_argument('-sl', '--snippet-length',
                                    dest='snippet_length',
                                    help='length of snippets for 3D splits in number of frames.',
                                    type=int, required=False, default=64)
    ap.add_argument('-sw', '--snippet-width',
                                    dest='snippet_width',
                                    help='time width of snippets for 3D splits in seconds.',
                                    type=int, required=False, default=5)
    ap.add_argument('-cf', '--contiguous-frames',
                                    dest='contiguous_frames',
                                    help='should be contiguous frames selected.',
                                    type=int, required=False, default=0)
    args = ap.parse_args()
    print(args)
    return args

def define_localization_label(labels, position):

    #the labels cames here sorted in reverse order.
    # so we can verify from the last of the video to the beggining if the possition \
    #fits a specific localization label in that piece of time.
    for label in labels:
        if position >= label[1][0]:
            if label[1][2] == 't':
                return 1
            else:
                return 0

def define_snippet_bounds(labels, position, frame_position, fps, frame_count, bound_unit, is_test_split, should_print):

    #the labels cames here sorted in reverse order.
    # so we can verify from the last of the video to the beggining if the possition \
    #fits a specific localization label in that piece of time.

    #We will verify the last label first and check if last portion of video is inside frame_count
    if is_test_split:
        should_print = False
        snippet_begin = 0
        snippet_end = int(min(frame_count -1, (labels[0][1][0] + labels[0][1][1]) * fps))
        if(should_print):
            print('position', position, 'last_label_init', labels[0][1][0], 'last_label_end', labels[0][1][1], frame_position, frame_count)
            print('ZZZZZZZZZZZZZZZ', snippet_begin, snippet_end )
        return snippet_begin, snippet_end

    if(should_print):
        print('position', position, 'last_label_init', labels[0][1][0], 'last_label_end', labels[0][1][1], frame_position, frame_count)
    if (position >= labels[0][1][0] and frame_position <= frame_count):
        if(should_print):
            print('11111111111111')

        if ('frame' in bound_unit):
            snippet_begin = int(math.floor(labels[0][1][0] * fps))
            snippet_end = int(min(frame_count -1, (labels[0][1][0] + labels[0][1][1]) * fps))
            if(should_print):
                print('222222222222222', snippet_begin, snippet_end )
        return snippet_begin, snippet_end

    for label in labels:
        snippet_begin = label[1][0]
        snippet_end = label[1][0] + label[1][1]
        if round(position,3) >= snippet_begin and position <= snippet_end:
            if ('frame' in bound_unit):
                snippet_begin = int(math.floor(snippet_begin * fps))
                snippet_end = int(math.floor(snippet_end * fps))
                if(should_print):
                    print('33333333333333', snippet_begin, snippet_end )
            return snippet_begin, snippet_end

    raise ValueError('Error define_snippet_bounds for position {}'.format(position))


def calc_desired_bounds(frame_position, fps, args):
    snippet_begin = frame_position
    snippet_end = int(math.floor(frame_position + (args.snippet_width * fps)))
    return snippet_begin, snippet_end

def fit_bounds(min_init, max_end, desired_init, desired_end, args):
    end_bound = desired_end if desired_end <= max_end else max_end
    init_bound = desired_init if desired_init >= min_init else min_init

    return init_bound, end_bound

def frange(x, y, jump):
  while x < y:
    yield int(round(x))
    x += jump

def select_snippet_frames(init_bound, end_bound, fps, args):
    snippet = []
    window = end_bound - init_bound

    step = 1.0
    if not args.contiguous_frames:
        step = float(window / args.snippet_length)
        step = step if step > 0 else 1.0

    while len(snippet) < args.snippet_length:
        for frame in list(frange(init_bound, end_bound, decimal.Decimal(str(step)))):
            snippet.append(frame)
        #lets change a bit the frames if needed to loop again by summing 1 on init_bound if possible
        init_bound = init_bound + 1 if init_bound < end_bound -1 else init_bound
    return snippet[:args.snippet_length]

#vPorn000235.mp4 1 296.858 4.0107 event - porn - f
def get_localization_labels(filename):
    import operator
    i = 0
    content = {}

    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        values = line.split(' ')
        content[i] = [float(values[2]), float(values[3]), values[8][:1]]
        i += 1

    sorted_labels = sorted(content.items(), key=operator.itemgetter(0), reverse = True )
    #print "sorted_labels :%s" % sorted_labels
    return sorted_labels

def is_snippet_length_too_short(init_bound, end_bound, fps, args):
    snippet_length = end_bound - init_bound
    desired_length = args.snippet_width * fps
    # print('snippet_length, desired_length', snippet_length, desired_length)
    # snippets fewer than 0.5 * desired length will be discarded
    if snippet_length <=0:
        print('*', end='')
        return True 
    if snippet_length < 0.3 * desired_length:
        print('!', end='')
        return True
    # if snippet_length < 0.1 * args.snippet_length:
    #     print('@', end='')
    #     return True
    print('.', end='')
    return False

def generate_snippet(video_name,frame_entry, split_type, frame_count, fps, etf_file, labels, frame_position, position, args, is_test_split):

    should_print = False
    if video_name in('vPorn001000', 'vPorn000316'):
        should_print = True
        print(frame_entry, frame_count, fps, frame_position, position, labels)
    min_init, max_end = define_snippet_bounds(labels, position, frame_position, fps, frame_count, 'frame', is_test_split, should_print)
    if video_name in('vPorn001000', 'vPorn000316'):
        print('max', min_init, max_end)
    desired_init, desired_end = calc_desired_bounds(frame_position, fps, args)
    if video_name in('vPorn001000', 'vPorn000316'):
        print('desired', desired_init, desired_end)
    init_bound, end_bound = fit_bounds(min_init, max_end, desired_init, desired_end, args)
    if video_name in('vPorn001000', 'vPorn000316'):
        print('fit', init_bound, end_bound)

    #If the length of snippet is to low we will not create it
    if(is_snippet_length_too_short(init_bound, end_bound, fps, args)):
        return False
    #print('|', end='', flush=True)
    frames_snippet = select_snippet_frames(init_bound, end_bound, fps, args)
    #print('$', end='', flush=True)

    snippet_folder = os.path.join(args.output_path, args.split_number, split_type, '{}_fps'.format(args.sample_rate), 'w_{}_l_{}'.format(args.snippet_width, args.snippet_length), video_name)
    if not os.path.exists(snippet_folder):
            os.makedirs(snippet_folder)

    snippet_path = os.path.join(snippet_folder, frame_entry + '.txt')
    with open(snippet_path, "w") as f:
        for item in frames_snippet:
                f.write("%s\n" % item)
    return True

def get_frame_count(video_name, args):
  frame_count_path = os.path.join(args.dataset_dir, 'etf_frame_count', '{}.etf'.format(video_name))
  with open(frame_count_path, "r") as f:
    length = int(f.read())

  return length

def select_video_frames(video_name, split_type, args, split_test):
    print('\n', video_name, ' ', end='')
    frames = []
    etf_file = os.path.join(args.dataset_dir, 'etf', video_name + '.etf')
    _, fps, _, _ = opencv.get_video_params(os.path.join(args.dataset_dir, 'videos', video_name + '.mp4'))
    frame_count = get_frame_count(video_name, args)
    calc_duration = (frame_count * 1000)/fps

    labels = get_localization_labels(etf_file)
    etf_duration = (labels[0][1][0] + labels[0][1][1]) * 1000 

    # duration_difference = calc_duration - etf_duration
    # if abs(duration_difference) > 1000 and 'vPorn' in video_name: # > 2000 == 2 seconds
    #     print(video_name, '{0:0.2f} {1}'.format(abs(duration_difference), '+' if duration_difference > 0 else '-'))
    for frame_position in list(frange(0, frame_count, decimal.Decimal(str(fps * args.sample_rate)))):
    # for frame_position in range (0, frame_count, int(math.floor(fps * args.sample_rate)) ):
        position = float(frame_position) / float(fps)
        localization_label = 0 if 'vNonPorn' in video_name else define_localization_label(labels, position)
        frame_entry = "{}_{}_{}".format(video_name, localization_label, frame_position)
        #for 3D case, we will append the frames only if the snippet is really generated
        if ('3D' in split_type):
            #print('.', end='', flush=True)
            if(generate_snippet(video_name,frame_entry, split_type, frame_count, fps, etf_file, labels, frame_position, position, args, split_test)):
                frames.append(frame_entry)
        else:
            frames.append(frame_entry)

    return frames

def create_video_split(name_set, split_type, positive_set, negative_set, args, split_test=False):
    split = []
    all_set = positive_set + negative_set
    for video in all_set:
        split.extend(select_video_frames(video.strip(), split_type, args,split_test))


    split_path = os.path.join(args.output_path, args.split_number, split_type, '{}_fps'.format(args.sample_rate), name_set + '.txt')
    with open(split_path, "w") as f:
        for item in split:
                f.write("%s\n" % item)

def create_splits(args):
    positive_network_validation_set = []
    negative_network_validation_set = []
    positive_network_training_set = []
    negative_network_training_set = []

    positive_svm_validation_set = []
    negative_svm_validation_set = []
    positive_svm_training_set = []
    negative_svm_training_set = []

    positive_test_set = []
    negative_test_set = []

    full_dir_path = os.path.join(args.dataset_dir, 'folds', args.split_number)

    ###########################################
    #collecting all split1 training videos

    positive_network_training_set_path = os.path.join(full_dir_path, 'positive_network_training_set.txt')
    negative_network_training_set_path = os.path.join(full_dir_path, 'negative_network_training_set.txt')
    with open(positive_network_training_set_path) as f:
        positive_network_training_set = f.readlines()
    with open(negative_network_training_set_path) as f:
        negative_network_training_set = f.readlines()

    ###########################################
    #collecting all split1 validation videos

    positive_network_validation_set_path = os.path.join(full_dir_path, 'positive_network_validation_set.txt')
    negative_network_validation_set_path = os.path.join(full_dir_path, 'negative_network_validation_set.txt')
    with open(positive_network_validation_set_path) as f:
        positive_network_validation_set = f.readlines()
    with open(negative_network_validation_set_path) as f:
        negative_network_validation_set = f.readlines()
        

    ###########################################
    #collecting all split1 svm training videos

    positive_svm_training_set_path = os.path.join(full_dir_path, 'positive_svm_training_set.txt')
    negative_svm_training_set_path = os.path.join(full_dir_path, 'negative_svm_training_set.txt')
    with open(positive_svm_training_set_path) as f:
        positive_svm_training_set = f.readlines()
    with open(negative_svm_training_set_path) as f:
        negative_svm_training_set = f.readlines()

    ###########################################
    #collecting all split1 svm validation videos

    positive_svm_validation_set_path = os.path.join(full_dir_path, 'positive_svm_validation_set.txt')
    negative_svm_validation_set_path = os.path.join(full_dir_path, 'negative_svm_validation_set.txt')
    with open(positive_svm_validation_set_path) as f:
        positive_svm_validation_set = f.readlines()
    with open(negative_svm_validation_set_path) as f:
        negative_svm_validation_set = f.readlines()
        
    ###########################################
    #collecting all split1 test videos

    positive_test_set_path = os.path.join(args.dataset_dir, 'folds', args.split_number + '_negative_test.txt')
    negative_test_set_path = os.path.join(args.dataset_dir, 'folds', args.split_number + '_positive_test.txt')

    with open(positive_test_set_path) as f:
        positive_test_set = f.readlines()
    with open(negative_test_set_path) as f:
        negative_test_set = f.readlines()


    full_dir_path = os.path.join(args.output_path, args.split_number, '3D', '{}_fps'.format(args.sample_rate), 'w_{}_l_{}'.format(args.snippet_width, args.snippet_length))
    command = "mkdir -p " + full_dir_path
    print('\n', command)
    call(command, shell=True)

    create_video_split('network_training_set', '3D', positive_network_training_set, negative_network_training_set, args)
    create_video_split('network_validation_set', '3D', positive_network_validation_set, negative_network_validation_set, args)
    create_video_split('svm_training_set', '3D', positive_svm_training_set, negative_svm_training_set, args)
    create_video_split('svm_validation_set', '3D', positive_svm_validation_set, negative_svm_validation_set, args)

    print('################################ TEST SET ####################################')

    create_video_split('test_set', '3D', positive_test_set, negative_test_set, args, split_test=True)
    
    sets_3d_path = os.path.join(args.output_path, args.split_number, '3D', '{}_fps'.format(args.sample_rate))

    sets_2d_path = os.path.join(args.output_path, args.split_number, '2D', '{}_fps'.format(args.sample_rate))
    command = "mkdir -p " + sets_2d_path
    print('\n', command)
    call(command, shell=True)
    command = 'cp {}/*.txt {}'.format(sets_3d_path, sets_2d_path)
    print('\n', command)
    call(command, shell=True)

def main():
    print('> Create splits from videos -', time.asctime( time.localtime(time.time())))
    args = load_args()
    create_splits(args)
    print('\n> Create splits  from videos done -', time.asctime( time.localtime(time.time())))
    return 0


if __name__ == '__main__':
    main()