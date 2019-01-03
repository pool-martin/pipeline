from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import sys

if __name__ == '__main__':
    # checkpoint=unicode(str(sys.argv[1:][0]), "utf-8")
    # print checkpoint
    print_tensors_in_checkpoint_file(file_name=sys.argv[1:][0], tensor_name='', all_tensors=False)