#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  get_labels_violence.py
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
calculate labels to be used in mediaeval violence dataset. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com)
'''

import argparse, os, time, random, math
from subprocess import call
import re
from utils import opencv

global_labels = {}

def load_args():
    ap = argparse.ArgumentParser(description='Create Splits to be used in 2D or 3D CNN models.')
    ap.add_argument('-d', '--dataset-dir',
                                    dest='dataset_dir',
                                    help='path to dataset files.',
                                    type=str, required=False, default='/DL/violence/')
    ap.add_argument('-o', '--output-path',
                                    dest='output_path',
                                    help='path to output the extracted frames.',
                                    type=str, required=False, default='/Exp/violence/splits/')
    ap.add_argument('-s', '--split-number',
                                    dest='split_number',
                                    help='split to be created.',
                                    type=str, required=False, default='s1')
    ap.add_argument('-sr', '--sample-rate',
                                    dest='sample_rate',
                                    help='sample rate to be used in video frame sampling.',
                                    type=int, required=False, default=5)
    ap.add_argument('-sl', '--snippet-length',
                                    dest='snippet_length',
                                    help='length of snippets for 3D splits in number of frames.',
                                    type=int, required=False, default=125)
    ap.add_argument('-sw', '--snippet-width',
                                    dest='snippet_width',
                                    help='time width of snippets for 3D splits in seconds.',
                                    type=int, required=False, default=5)
    ap.add_argument('-mp', '--mediaeval-path',
                                    dest='mediaeval_path',
                                    help='path to videos and lists.',
                                    type=str, required=False, default='/mediaeval')

    args = ap.parse_args()
    print(args)
    return args

def select_video_frames(video_name, split_type, args, split_test):
    global global_labels
    labels_set = {}
    video, fragment, _ = re.split(r'[\-\.]', video_name)
#    print('\n', video, fragment)

    if not split_test:
        if video not in global_labels:
            video_labels_path = os.path.join(args.mediaeval_path, 'lists', 'frames_matrices', 'samples_png_{}_matrix.labels'.format(video))
            with open(video_labels_path, "r") as f:
                for line in f:
                    _, init_pos, _, l0, l1, l2, l3, l4, l5, l6, l7 = re.split(r'[\-\s]', line.strip())
                    labels_set[int(init_pos)] = [int(l0), int(l1), int(l2), int(l3), int(l4), int(l5), int(l6), int(l7)]

            global_labels[video] = labels_set
        else:
            labels_set = global_labels[video]


        l0 = 0
        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0
        l5 = 0
        l6 = 0
        l7 = 0
        initial_frame = 125 * int(fragment)
        for i in range(initial_frame, initial_frame + 125):
            l0 = l0 or labels_set[i][0]
            l1 = l1 or labels_set[i][1]
            l2 = l2 or labels_set[i][2]
            l3 = l3 or labels_set[i][3]
            l4 = l4 or labels_set[i][4]
            l5 = l5 or labels_set[i][5]
            l6 = l6 or labels_set[i][6]
            l7 = l7 or labels_set[i][7]
        
        result = '{} {} {} {} {} {} {} {} {}'.format(video_name, l0, l1, l2, l3, l4, l5, l6, l7)
    else:
        result = '{} {} {} {} {} {} {} {} {}'.format(video_name, 9, 9, 9, 9, 9, 9, 9, 9)
    print(result)

    return result

def get_labels_fragments(name_set, split_type, all_set, args, split_test=False):
    split = []
#    video_id = ""
    for video in all_set:
#        video_id, _, _ = re.split(r'[\-\.]', video)
#        if video_id not in ['FAN4', 'FARG', 'FORG', 'LEGB', 'PULP', 'TGOD', 'TPIA']:
        split.append(select_video_frames(video.strip(), split_type, args, split_test))

    split_path = os.path.join(args.output_path, args.split_number, '3D', '{}_fps'.format(args.sample_rate), name_set + '.txt')
    with open(split_path, "w") as f:
        for item in split:
                f.write("%s\n" % item)

def create_splits(args):
    network_validation_set = []
    network_training_set = []
    test_set = []

    full_dir_path = os.path.join(args.dataset_dir, 'folds')

    ###########################################
    #collecting all split1 training videos

    network_training_set_path = os.path.join(full_dir_path, 'training.txt')
    with open(network_training_set_path) as f:
        network_training_set = f.readlines()

    ###########################################
    #collecting all split1 validation videos

    network_validation_set_path = os.path.join(full_dir_path, 'validation.txt')
    with open(network_validation_set_path) as f:
        network_validation_set = f.readlines()
        

    ###########################################
    #collecting all split1 test videos

    test_set_path = os.path.join(full_dir_path, 'test.txt')
    with open(test_set_path) as f:
        test_set = f.readlines()

    full_dir_path = os.path.join(args.output_path, args.split_number, '3D', '{}_fps'.format(args.sample_rate))
    command = "mkdir -p " + full_dir_path
    print('\n', command)
    call(command, shell=True)


    get_labels_fragments('network_training_set', 'train', network_training_set, args)
    get_labels_fragments('network_validation_set', 'val', network_validation_set, args)

    print('################################ TEST SET ####################################')

    get_labels_fragments('test_set', 'test', test_set, args, split_test=True)
    

def main():
    print('> Create splits from videos -', time.asctime( time.localtime(time.time())))
    args = load_args()
    create_splits(args)
    print('\n> Create splits  from videos done -', time.asctime( time.localtime(time.time())))
    return 0


if __name__ == '__main__':
    main()