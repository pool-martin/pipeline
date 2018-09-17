import os, sys
import pickle

import utils.helpers as helpers
_PREDICTION_OUTPUT_FORMAT='%.16f'

def save_extracted_features(FLAGS, set_name, set_to_extract, pred_generator):

    print('Extracting {} set'.format(set_name))
    outfile = open(helpers.assembly_extract_features_filename(FLAGS, set_name), 'w')
    num_outputs = len(set_to_extract)
    if FLAGS.output_format=='text' :
        print(num_outputs, file=outfile)
        header  = [ 'snippet_id' ]
        header += [ 'truth' ]
#        header += [ 'feature[%d]' % feature_size ]
        print(', '.join(header), file=outfile)
#    else :
#        pickle.dump([num_outputs, feature_size, FLAGS.__flags], outfile)

#    features_size = list_sample[0]["features"].get_shape()[0]
#    pickle.dump([len(list_sample), features_size, FLAGS.__flags], outfile)

    s = 0
    for sample in pred_generator:
        print('{', end='', file=sys.stderr)
        snippet_id = sample['snippet_id']
        label = sample['truth_label']
        feats = sample['features']
        if FLAGS.output_format=='text' :
            record  = [ snippet_id.decode("utf-8") ]
            record += [ str(label) ]
            record += [ _PREDICTION_OUTPUT_FORMAT % feats[f]  for f in range(feats.shape[0]) ]
            print(', '.join(record), file=outfile)
        else :
            pickle.dump([snippet_id, label, feats], outfile)
        s += 1
        print('}', end='\n' if (s+1) % 40 == 0 else '', file=sys.stderr)
    print('', file=sys.stderr)
