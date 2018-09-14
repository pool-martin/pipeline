import os

def assembly_sets_path(FLAGS):
    if FLAGS.force_splits_dir_path:
        sets_path = FLAGS.sets_dir
    else:
        sets_path = os.path.join(FLAGS.sets_dir, FLAGS.split_number, FLAGS.split_type, '{}_fps'.format(FLAGS.sample_rate) )
    return sets_path

def assembly_snippets_path(FLAGS):
    if FLAGS.force_splits_dir_path:
        snippets_path = FLAGS.snippets_dir
    else:
        snippets_path = os.path.join(FLAGS.sets_dir, FLAGS.split_number, FLAGS.split_type, '{}_fps'.format(FLAGS.sample_rate), 'w_{}_l_{}'.format(FLAGS.snippet_width, FLAGS.snippet_size) )
    return snippets_path

def assembly_model_dir(FLAGS):
    model_dir_path = os.path.join(FLAGS.model_dir, FLAGS.model_name, FLAGS.experiment_tag)
    return model_dir_path 