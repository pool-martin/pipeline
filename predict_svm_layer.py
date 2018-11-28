# Copyright 2017 Eduardo Valle. All rights reserved.
# eduardovalle.com/ github.com/learningtitans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import pickle
import sys
import os

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing

from svm_layer import utils as su

os.environ['JOBLIB_TEMP_FOLDER'] = "~/tmp"

def read_pmsvm_data(input_training):
    ids =[]
    num_samples = 0
    with open(input_training, 'r') as file:
        num_samples = sum(1 for _ in file)
    print('num_samples :',  num_samples, file=sys.stderr)
#    labels = np.empty([251616], dtype=np.float)
#    features = np.empty([251616, 1024], dtype=np.float)
#    labels = np.empty([258915], dtype=np.float)
#    features = np.empty([258915, 1024], dtype=np.float)
    labels = np.empty([num_samples], dtype=np.float)
    features = np.empty([num_samples, 1024], dtype=np.float)
#    labels = {}
#    features = {}
    i = 0
    with open(input_training, 'r') as f:
        for line in f:
            feature_dic = {}
            ids.append(i)
            feature_line = line.split(' ')
            label = int(feature_line.pop(0))
            if '\n' == feature_line[-1]:
#                feature_line = feature_line[:-1]
                feature_line.pop(-1)
            labels[i] = label
            
            for column in feature_line:
                column_split = column.split(':')
                index = long(column_split[0])
                value = float(column_split[1])
                feature_dic[index] = value
            row = np.empty([1024], dtype=np.float)
            for j in range(1, 1025):
                value = feature_dic.get(j, 0.)
                if np.isnan(value):
                    row[j-1] = 0
                else:
                    row[j-1] = value
            features[i] = row
            if(i % 1000 == 0):
                print('reading i=', i, file=sys.stderr)
            i += 1
            
    return ids, labels, features

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--input_model', type=str, required=True, help='input trained model, in pickle format.')
parser.add_argument('--input_test', type=str, required=True, help='input file with the test data, in pickle format.')
parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
parser.add_argument('--output_metrics', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--output_images', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--pool_by_id', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
parser.add_argument('--compute_rolling_window', default=False, action='store_true', help='compute rolling_window')
parser.add_argument('--reading_from_libsvm', default=False, action='store_true', help='reading from libsvm_format')
parser.add_argument('--video_split_char', type=str, default='.', help='char to split video name')

FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)
model_file = open(FLAGS.input_model, 'rb')
preprocessor = pickle.load(model_file)
classifier_m = pickle.load(model_file)
#classifier_k = pickle.load(model_file)
model_file.close()

start = su.print_and_time('Reading test data...',  past=start, file=sys.stderr)
if FLAGS.reading_from_libsvm:
    image_ids, labels, features = read_pmsvm_data(FLAGS.input_test)
else:
    image_ids, labels, features = su.read_pickled_data(FLAGS.input_test)
num_samples = len(image_ids)

start = su.print_and_time('Preprocessing test data...', past=start, file=sys.stderr)
features = preprocessor.transform(features)

# "Probabilities" should come between quotes here
# Only if the scores are true logits the probabilities will be consistent
def probability_from_logits(logits) :
    odds = np.exp(logits)
    return odds/(odds+1.0)
def logits_from_probability(prob) :
    with np.errstate(divide='ignore') :
      odds = prob/(1.0-prob)
      return np.log(odds)
def extreme_probability(prob) :
  return prob[np.argmax(np.abs(logits_from_probability(prob)))]

start = su.print_and_time('Predicting test data...\n', past=start, file=sys.stderr)
confidence_scores_m = classifier_m.decision_function(features)
#confidence_scores_k = classifier_k.decision_function(features)

predictions_m = probability_from_logits(confidence_scores_m)
#predictions_k = probability_from_logits(confidence_scores_k)


outfile = open(FLAGS.output_predictions, 'w') if FLAGS.output_predictions else sys.stdout
if FLAGS.pool_by_id=='none' :
  for i in range(len(image_ids)) :
#    print(image_ids[i], predictions_m[i], predictions_k[i], confidence_scores_m[i], confidence_scores_k[i], sep=',', file=outfile)
    print(image_ids[i].decode("utf-8"), labels[i], predictions_m[i], confidence_scores_m[i], sep=',', file=outfile)
else :
  previous_id = None
  def print_result() :
    if FLAGS.pool_by_id=='avg' :
#      print(previous_id, np.mean(all_m), np.mean(all_k), sep=',', file=outfile)
      print(previous_id, np.mean(all_m), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='max' :
#      print(previous_id, np.amax(all_m), np.amax(all_k), sep=',', file=outfile)
      print(previous_id, np.amax(all_m), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='xtrm' :
#      print(previous_id, extreme_probability(all_m), extreme_probability(all_k), sep=',', file=outfile)
      print(previous_id, extreme_probability(all_m), sep=',', file=outfile)
    else :
      raise ValueError('Invalid value for FLAGS.pool_by_id: %s' % FLAGS.pool_by_id)

  for i in range(len(image_ids)) :
    if image_ids[i]!=previous_id :
      if previous_id is not None :
        print_result()
      previous_id = image_ids[i]
      all_m = np.asarray([ predictions_m[i] ])
#      all_k = np.asarray([ predictions_k[i] ])
    else :
      all_m = np.concatenate((all_m, np.asarray([ predictions_m[i] ])))
#      all_k = np.concatenate((all_k, np.asarray([ predictions_k[i] ])))
  if previous_id is not None :
    print_result()

outfile.close()
metfile = open(FLAGS.output_metrics, 'w') if FLAGS.output_metrics else sys.stderr
try :
  accs = []
  aucs_s = []
  aucs_p = []
  mAPs_s = []
  mAPs_p = []
#  for j, scores_j, predictions_j in [ [1., confidence_scores_m, predictions_m], [0., confidence_scores_k, predictions_k] ] :
  for j, scores_j, predictions_j in [ [1., confidence_scores_m, predictions_m]] :
    labels_j = (labels == j).astype(np.int)
    acc = sk.metrics.accuracy_score(labels_j, (predictions_j >= .5).astype(np.int))
    print('Binary Acc: ', acc, file=metfile)
    accs.append(acc)
    auc = sk.metrics.roc_auc_score(labels_j, scores_j)
    aucs_s.append(auc)
    print('AUC scores [%d]: ' % j, auc, file=metfile)
    auc = sk.metrics.roc_auc_score(labels_j, predictions_j)
    aucs_p.append(auc)
    print('AUC predictions [%d]: ' % j, auc, file=metfile)
    mAP = sk.metrics.average_precision_score(labels_j, scores_j)
    mAPs_s.append(mAP)
    print('mAP scores [%d]: ' % j, mAP, file=metfile)
    mAP = sk.metrics.average_precision_score(labels_j, predictions_j)
    mAPs_p.append(mAP)
    print('mAP predictions [%d]: ' % j, mAP, file=metfile)
#  print('Acc_avg: ', sum(accs) / 2.0, file=metfile)
#  print('AUC_s_avg: ', sum(aucs_s) / 2.0, file=metfile)
#  print('AUC_p_avg: ', sum(aucs_p) / 2.0, file=metfile)
#  print('mAP_s_avg: ', sum(mAPs_s) / 2.0, file=metfile)
#  print('mAP_p_avg: ', sum(mAPs_p) / 2.0, file=metfile)

  def balanced_acc(tn, fp, fn, tp):
    TPR = tp/(tp+fn)
    TNR = tn/(tn+fp)
    Balanced_Acc = (TPR + TNR)/2
    print('tn: ', tn, 'fp: ', fp, 'fn: ', fn, 'tp: ', tp, 'recal(TPR): ', TPR, 'TNR: ', TNR, 'Balanced_acc: ', Balanced_Acc,  file=metfile)
    return Balanced_Acc

  def harmonic_acc(tn, fp, fn, tp):
    TPR = tp/(tp+fn)
    TNR = tn/(tn+fp)
    Harmonic_Acc = (2 * (TPR * TNR))/(TPR + TNR)
    print('tn: ', tn, 'fp: ', fp, 'fn: ', fn, 'tp: ', tp, 'recal(TPR): ', TPR, 'TNR: ', TNR, 'Harmonic_Acc: ', Harmonic_Acc,  file=metfile)
    return Harmonic_Acc

  #Balanced acc
#  acc = sk.metrics.accuracy_score(labels, (predictions_m >= 0.5).astype(np.int))
  tn, fp, fn, tp = sk.metrics.confusion_matrix((labels == 1.).astype(np.int), (predictions_m >= 0.5).astype(np.int)).ravel()
  Balanced_Acc = balanced_acc(tn, fp, fn, tp)
  print('Balanced Acc (predictions): ', Balanced_Acc, file=metfile)

#  false_positive_rate, true_positive_rate, thresholds = sk.metrics.roc_curve(labels, (predictions_m >= predictions_k).astype(np.int))
  false_positive_rate, true_positive_rate, thresholds = sk.metrics.roc_curve((labels == 1.).astype(np.int), predictions_m)
  print('labels predictions len: ', len(labels), len(predictions_m), file=sys.stdout)
  print('fpr tpr len: ', len(false_positive_rate), len(true_positive_rate), file=sys.stdout)
  roc_auc = sklearn.metrics.auc(false_positive_rate, true_positive_rate)
  #image_name = FLAGS.output_metrics.rsplit('/',1)[1].split('.')
  image_name = FLAGS.output_metrics.rsplit('/',1)[1]
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  plt.title('ROC curve ' + image_name)
  plt.plot(false_positive_rate, true_positive_rate, 'b',
  label='AUC = %0.4f'% roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
#  plt.xlim([-0.1,1.2])
#  plt.ylim([-0.1,1.2])
  plt.xlim([0.0,1.0])
  plt.ylim([0.0,1.0])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
#  plt.show()
#  plt.savefig("/data/tf/1_s/running/svm.predictions/"+ image_name[0] + "." + image_name[1] +".roc.png")
  plt.savefig(FLAGS.output_images +".roc.png")
  
  if FLAGS.compute_rolling_window:
    print('\n Starting pandas stuff', end='', file=sys.stderr)
    import pandas as pd

    #  df = pd.read_csv(FLAGS.output_predictions, names=['Frame', 'prob_porn', 'prob_nonporn', 'score_porn', 'score_nonporn'])
    df = pd.read_csv(FLAGS.output_predictions, names=['Frame', 'previous_labels', 'prob_porn', 'score_porn'])
    print('\n Could read the csv', end='', file=sys.stderr)
    df = df.sort_values(by='Frame')
    print('\n Sorted by frame', end='', file=sys.stderr)
    def compare(row):
        if row['prob_porn'] >= 0.5:
            return 1
        else:
            return -1
    def compare2(row):
        return row['prob_porn'] - 0.5
    print('\n Created prediction', end='', file=sys.stderr)

    df['predictions'] = df.apply(compare, axis=1)
    df['prob_porn_2'] = df.apply(compare2, axis=1)

    #def extract_label(row):
    #    return int(row['Frame'].split('.')[3])
    #df['labels'] = df.apply(extract_label, axis=1)

    #print('\n Created labels', end='', file=sys.stderr)

    def extract_video_name(row):
        return row['Frame'].split(FLAGS.video_split_char)[0]
    df['videos'] = df.apply(extract_video_name, axis=1)
    print('\n Created videos', end='', file=sys.stderr)

    def rolling_window(df, k, shape, column):
#        if shape == 'square':
#            new_column = ((df.groupby('videos', sort=False)[column].rolling(k, center=True, min_periods=1).sum() >= 0).astype(int))
#            if k == 3:
#                print(new_column)
#            df['k_pred_' + shape[0] + str(k)] = new_column.reset_index(level=0, drop=True)
#            if k == 3:
#                print(df['k_pred_' + shape[0] + str(k)])
#        elif shape == 'triangle':
#            new_column_t = ((df.groupby('videos', sort=False)[column].rolling(k, center=True, min_periods=1, win_type='triang').sum() >=  0).astype(int))
#            if k == 3:
#                print(new_column_t)
#            df['k_pred_' + shape[0] + str(k)] = new_column_t.reset_index(level=0, drop=True)
#            if k == 3:
#                print(df['k_pred_' + shape[0] + str(k)])
        groups = df.groupby('videos', sort=False)
        new_column = []
        gaussian_result = []
        sigma = k
        gaussianWindow = 6 * int(sigma) + 1
        for group_name, group_df in groups:
            if(shape == 'gaussian'):
                import cv2
                gaussian_result.extend(cv2.GaussianBlur(group_df[column].values, (gaussianWindow, gaussianWindow), k))
                #new_column.extend((group_df[column].rolling(k, center=True, min_periods=1, win_type=shape).mean(std=sigma)).values)
            else:
                new_column.extend((group_df[column].rolling(k, center=True, min_periods=1, win_type=shape).sum() >= 0).values)
        if(shape == 'gaussian'):
            for i in gaussian_result:
                new_column.extend(i >=0)
            df['k_' + column[:4] + '_' + shape[0] + str(k) + '_p'] = gaussian_result
        df['k_' + column[:4] + '_' + shape[0] + str(k)] = new_column

    def dict_to_pair(x):
        lists = sorted(x)
        x, y = zip(*lists)
        return x, y

    def save_graph(x,y,z,w, k, label, label_h, title_, image_name):
        #print('\n Salvar grafico: ', title, ' ', image_name, '\n', end='', file=sys.stderr)

        plt.clf()
        plt.title('Rolling Window ' + title_)
        plt.plot(x, y, label=label)
        plt.plot(z, w, label=label_h)
        plt.legend(loc='lower right')
        plt.xlim([0.0,k])
        plt.ylim([0.60,1.00])
        plt.ylabel('balanced acc')
        plt.xlabel('k')
        plt.savefig(FLAGS.output_images + ".rolling_window." + image_name + ".png")


    ######## Graph for b_acc_k_pred_s###############
    def run_for_experiment(range_k, column, window_shape, title, image_name):
        b_acc_k = {}
        h_acc_k = {}
        best_acc = 0.0
        best_k = 0
        best_h_acc = 0.0
        best_h_k = 0
        for i in range(1, range_k, 2):
            rolling_window(df, i, window_shape, column)

        print('\n Created k_labels: ', i, file=sys.stderr)

        df.to_csv(path_or_buf=FLAGS.output_predictions+".k_test")
        print('\n Saved dataframe', end='', file=sys.stderr)

        for k in range(1, range_k, 2):
            tn, fp, fn, tp = sk.metrics.confusion_matrix(df['previous_labels'].values, df['k_' + column[:4] + '_' + window_shape[0] +str(k)].values).ravel()
             
            b_acc = balanced_acc(tn, fp, fn, tp)
            #print('Acc k_', str(k) ,' ', b_acc, file=metfile)
            b_acc_k[k] = b_acc
            if b_acc > best_acc:
                best_acc = b_acc
                best_k = k
            h_acc = harmonic_acc(tn, fp, fn, tp)
            h_acc_k[k] = h_acc
            if h_acc > best_h_acc:
                best_h_acc = h_acc
                best_h_k = k
        x = {}
        y = {}
        x, y = dict_to_pair(b_acc_k.items())
        z = {}
        w = {}
        z, w = dict_to_pair(h_acc_k.items())
        label_b_acc = 'best => b_acc[' + str(best_k) + ']= ' + str(best_acc)
        label_h_acc = 'best => h_acc[' + str(best_h_k) + ']= ' + str(best_h_acc)
        save_graph(x,y,z,w, range_k,label_b_acc, label_h_acc, title, image_name)

    ######## Graph for b_acc_k_pred_t###############
    start = su.print_and_time('Will create acc predictions square...\n', past=start, file=sys.stderr)
    title = 'acc_k column prediction window square'
    image_name = 'predictions.square'
    #run_for_experiment(6, 'predictions', 'boxcar', title, image_name)

    start = su.print_and_time('\n Will create acc predictions triangle', past=start, file=sys.stderr)
    title = 'acc_k column prediction window triangle'
    image_name = 'predictions.triangle'
    #run_for_experiment(6, 'predictions', 'triang', title, image_name)

    start = su.print_and_time('\n Will create acc prob_porn square', past=start, file=sys.stderr)
    title = 'acc_k column prob_porn window square'
    image_name = 'prob_porn.square'
    #run_for_experiment(6, 'prob_porn_2', 'boxcar', title, image_name)

    start = su.print_and_time('\n Will create acc prob_porn triangle', past=start, file=sys.stderr)
    title = 'acc_k column prob_porn window triangle'
    image_name = 'prob_porn.triangle'
    #run_for_experiment(6, 'prob_porn_2', 'triang', title, image_name)

    start = su.print_and_time('\n Will create acc prob_porn gaussian', past=start, file=sys.stderr)
    title = 'acc_k column prob_porn window gaussian'
    image_name = 'prob_porn.gaussian'
    run_for_experiment(32, 'prob_porn_2', 'gaussian', title, image_name)
    start = su.print_and_time('\n Will create acc square over gaussian', past=start, file=sys.stderr)
    title = 'acc_k column k_prob_g5 window square'
    image_name = 'k_prob_g5.square'
#    run_for_experiment(6, 'k_prob_g5_p', 'boxcar', title, image_name)
    
#        tn, fp, fn, tp = sk.metrics.confusion_matrix(df['labels'].values, df['k_prob_s'+str(k)].values).ravel()
#        b_acc = balanced_acc(tn, fp, fn, tp)
#        #print('Acc k_', str(k) ,' ', b_acc, file=metfile)
#        b_acc_k_prob_s[k] = b_acc
#        tn, fp, fn, tp = sk.metrics.confusion_matrix(df['labels'].values, df['k_prob_t'+str(k)].values).ravel()
#        b_acc = balanced_acc(tn, fp, fn, tp)
#        #print('Acc k_', str(k) ,' ', b_acc, file=metfile)
#        b_acc_k_prob_t[k] = b_acc

#    lists = sorted(b_acc_k_pred_s.items())
#    x, y = zip(*lists)
    print('\n accs calculadas: ', i, file=sys.stderr)
    

    print('Printed Accs', file=sys.stderr)

    metfile.close()
  
except ValueError as e :
  print(e, file=sys.stderr)
  pass

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)