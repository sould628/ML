import code.CIFAR_INPUT as CIFAR_INPUT
import os
import tensorflow as tf

data_dir="./file/"

def main():
    filename_queue=[os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1,6)]
    labelFile=os.path.join(data_dir, 'batches.meta')
    for f in filename_queue:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)


    read_input=CIFAR_INPUT.read_CIFAR10(filename_queue, labelFile)


if __name__=="__main__":
    main()