import argparse
import pandas as pd
import sys
import os


def mount_result_etf(video_name, localization_flag, fps):
	#Let try to identify blocks of '0' or '1' and create a list of something like:
	# video_name #start_frame #number_of_frames #class
	current_snippet_class = localization_flag[0]
	classes_snippets = []
	current_frame_qty = 1
#	current_snippet_qty = 1
	current_snippet_initial_position = 0  #0 based
	for i in range (1, len(localization_flag)):
		if localization_flag[i] == current_snippet_class:
			current_frame_qty = current_frame_qty + 1
		else:
			beg = float(current_snippet_initial_position/fps)
			interval = float(i/fps) - beg
			classes_snippets.append([beg, interval, current_snippet_class])
#			current_snippet_qty = current_snippet_qty + 1
			current_snippet_class = localization_flag[i]
			current_snippet_initial_position = i
			current_frame_qty = 1
	beg = float(current_snippet_initial_position/fps)
	interval = float(i/fps) - beg
	classes_snippets.append([beg, interval, current_snippet_class])
	
	str_snippet_classes = ""
	for i in range (len(classes_snippets)):
		str_snippet_classes = str_snippet_classes + video_name + ".mp4 1 " + \
		str(classes_snippets[i][0]) + " " + str(classes_snippets[i][1]) + " event - porn - " + ('t' if classes_snippets[i][2] else 'f')  + "\r\n"
		#print str_snippet_classes
	
	return str_snippet_classes
	

def result_2_etf(df, is_3d, fps_sampled, result_row, FLAGS):

	localization_flag = {}
	#print(df)
	#clean list
	output_dir = os.path.join(FLAGS.output_path, FLAGS.set_to_process)
	result_etf_list = os.path.join(output_dir, "etf_list.txt")

	for index, row in df.iterrows():
		video_name = row['videos']
		result_etf = os.path.join(output_dir, video_name + ".etf")
		
		video_fps = os.path.join('/DL/2kporn', "video_fps", video_name + ".etf")
		video_length = os.path.join('/DL/2kporn', 'number_of_frames_video', video_name + ".etf")
		#print(video_fps)

		with open(video_fps, "r") as f:
			fps = float(f.read())
		#print(fps)
		with open(video_length, "r") as f:
			length = int(f.read())

		#Calc Localization flag:
		if is_3d:
			if row['Frame'].count('_') == 2:
				_, beg, end = row['Frame'].split('_')
			else:
				_, beg = row['Frame'].split('_')
				end = int(beg) + FLAGS.sample_width * fps
			for i in range(int(beg), int(end)+1):
				localization_flag[i] = row[result_row]
		else:
			video, frame, f_fps, gt_label  = row['Frame'].split('.')
			beg = (int(frame)- 1) * fps + 1
			end = beg + int(fps)
			for i in range(int(beg), int(end)+1):
				if i > length:
					break
				localization_flag[i] = row[result_row]

	str_result_etf = mount_result_etf(video_name, localization_flag, fps)
	with open(result_etf, "w") as f:
		f.write(str_result_etf)
	with open(result_etf_list, "a") as f:
 			f.write("%s\n" % result_etf)
		
def concat_files(FLAGS, folds_dir, etf_dir, output_path):
	if 'test' == FLAGS.set_to_process:
		positive_set_name = os.path.join(folds_dir, '%s_positive_test.txt' % FLAGS.fold_to_process)
		negative_set_name = os.path.join(folds_dir, '%s_negative_test.txt' % FLAGS.fold_to_process)
	else:
		positive_set_name = os.path.join(folds_dir, FLAGS.fold_to_process, 'positive_%s_set.txt' % FLAGS.set_to_process)
		negative_set_name = os.path.join(folds_dir, FLAGS.fold_to_process, 'negative_%s_set.txt' % FLAGS.set_to_process)
	
	video_set = []
	with open(positive_set_name, 'r') as f:
		video_set = f.readlines()
	with open(negative_set_name, 'r') as f:
		video_set += f.readlines()
	
	all_txt = ''
	for video_name in video_set:
		video_etf_path = os.path.join(etf_dir, video_name.rstrip('\n') + ".etf")
		if(os.path.isfile(video_etf_path)):
			with open(video_etf_path, 'r') as f:
				all_txt += f.read()
		else:
			print('{} not found'.format(video_etf_path))

	if not os.path.isdir(output_path):
		os.makedirs(output_path) 

	all_txt_file = os.path.join(output_path, 'all.txt')
	with open(all_txt_file, 'w') as f:
		f.write(all_txt)

def main():
    parser = argparse.ArgumentParser(prog='results_2_etf.py', description='create etfs based on results.')
    parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
    parser.add_argument('--output_path', type=str , help='folder to save the etf files.')
    parser.add_argument('--is_3d', default=False, action='store_true', help='results from 3d net?')
    parser.add_argument('--fps_sampled', type=int, default=1, help='fps used to sample')
    parser.add_argument('--set_to_process', type=str, default='svm_validation', help='Wich set should be processed for example svm_validation, test')
    parser.add_argument('--fold_to_process', type=str, default='s1', help='Wich fold should be processed for example s1, s2, ...')
    parser.add_argument('--column', type=str, default='k_prob_t5', help='Wich column to extract results, k_prob_t5, k_prob_t3, k_pred_t5, ...')
    parser.add_argument('--sample_width', type=int, default=1, help='How much time (seconds) the sample cover, ...')

    FLAGS = parser.parse_args()

    output_dir = os.path.join(FLAGS.output_path, FLAGS.set_to_process)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 


#    df = pd.read_csv(FLAGS.output_predictions+".k_test", names=['index','Frame', 'previous_labels', 'prob_porn', 'score_porn', 'predictions', 'prob_porn_2', 'videos', 'k_pred_b1', 'k_pred_b3', 'k_pred_b5', 'k_pred_t1', 'k_pred_t3', 'k_pred_t5'])
    df = pd.read_csv(FLAGS.output_predictions+".k_test")
    #print('\n Could read the csv', end='', file=sys.stderr)
    df = df.sort_values(by='Frame')
    #print('\n Sorted by frame', end='', file=sys.stderr)
	#clear list 
    output_dir = os.path.join(FLAGS.output_path, FLAGS.set_to_process)
    result_etf_list = os.path.join(output_dir, "etf_list.txt")
    open(result_etf_list, 'w').close()

    groups = df.groupby('videos', sort=False)
    for _, group_df in groups:
        result_2_etf(group_df, FLAGS.is_3d, FLAGS.fps_sampled, FLAGS.column, FLAGS)

    concat_files(FLAGS, '/DL/2kporn/folds', '/DL/2kporn/etf', os.path.join(output_dir, 'ground_truth'))
    concat_files(FLAGS, '/DL/2kporn/folds', output_dir, output_dir)

if __name__ == '__main__':
	main()
