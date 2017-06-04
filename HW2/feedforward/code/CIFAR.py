import tensorflow as tf
import numpy as np
import os
import code.CIFAR_INPUT as CIFAR_INPUT

#Constants for Training Process
MOVING_AVERAGE_DECAY=0.9999
NUM_EPOCHS_PER_DECAY=350.0
LEARNING_RATE_DECAY_FACTOR=0.1
INITIAL_LEARNING_RATE=0.1
MAX_EPOCH=100

tf.app.flags.DEFINE_integer('batch_size', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './../file/', """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('hidden1', 1024, """Number of 1st hidden units""")
tf.app.flags.DEFINE_integer('hidden2', 384, """Number of 2nd hidden units""")
tf.app.flags.DEFINE_integer('hidden3', 192, """Number of 3rd hidden units""")
tf.app.flags.DEFINE_float('learning_rate', INITIAL_LEARNING_RATE, """Learning Rate""")
tf.app.flags.DEFINE_string('log_dir', './../log/','''Path to the log''')
tf.app.flags.DEFINE_integer('max_epoch', MAX_EPOCH, """MAXSTEP""")

FLAGS=tf.app.flags.FLAGS

#GLOBAL CONSTANTS for CIFAR-10 data set.
IMAGE_SIZE=CIFAR_INPUT.IMAGE_SIZE
IMAGE_PIXELS=CIFAR_INPUT.IMAGE_PIXELS
NUM_CLASSES=CIFAR_INPUT.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


def placeholder_inputs(batch_size):
    """Generate placeholder variables for input tensors"""
    """Fed from the data in the .run() loop
    
    Input: Batch_Size: user defined batchsize (-1 for automatic set)"""
    images_placeholder=tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS))
    labels_placeholder=tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
    return images_placeholder, labels_placeholder




def inference(images, hidden1_units, hidden2_units, hidden3_units):
    """Build CIFAR-10 Model"""
    """Input
        images: Image_Placeholder
        hiddenX_units: size of the X-th hidden layer
        Output: Logits"""
    import math
    #hidden1
    with tf.name_scope('hidden1'):
        w1=tf.get_variable(
            'w1',
            shape=[IMAGE_PIXELS, hidden1_units],
            initializer=tf.contrib.layers.xavier_initializer()
            )
        b1=tf.get_variable('b1', shape=[hidden1_units], initializer=tf.zeros_initializer())
        hidden1=tf.nn.tanh(tf.matmul(images, w1)+b1)

    #hidden2
    with tf.name_scope('hidden2'):
        w2=tf.get_variable(
            'w2',
            shape=[hidden1_units, hidden2_units],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b2=tf.get_variable('b2', shape=[hidden2_units], initializer=tf.zeros_initializer())
        hidden2=tf.nn.tanh(tf.matmul(hidden1, w2)+b2)
    #hidden3
    with tf.name_scope('hidden3'):
        w3=tf.get_variable(
            'w3',
            shape=[hidden2_units, hidden3_units],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b3=tf.get_variable('b3', shape=[hidden3_units], initializer=tf.zeros_initializer())
        hidden3=tf.nn.tanh(tf.matmul(hidden2, w3)+b3)

    with tf.name_scope('softmax_linear'):
        wo=tf.get_variable(
            'wo',
            shape=[hidden3_units, NUM_CLASSES],
            initializer=tf.contrib.layers.xavier_initializer()
            )
        bo=tf.get_variable('bo', shape=[NUM_CLASSES], initializer=tf.zeros_initializer())
        logits=tf.matmul(hidden3, wo) + bo
        output_layer=tf.nn.softmax(logits)
        print(logits)
        print("output_layer", output_layer)
    return logits, output_layer

def loss(logits, label_one_hot):
    """Calculate the loss from the logits and the labels"""
    """
    Input
    logits: Logits tensor, float ([batch_size, NUM_CLASSES])
    labels: Labels tensor, int32 ([batch_size]
    
    output
    loss: Loss tensor of type float
    """
    output_layer=tf.nn.softmax(logits)
    loss=tf.reduce_mean(-tf.reduce_sum(label_one_hot*tf.log(output_layer), axis=1))
    return loss

def training(loss, learning_rate):
    """Sets up the training
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by is what must be passed to the 'sess.run()' call to cause the model to train"""
    """
    Input
    loss: Loss tensor from loss() function
    learning_rate: learning rate to use for gradient descent
    
    Output:
    train_op: the Op for training"""

    #Scalar summary for loss
    tf.summary.scalar('loss', loss)
    #Gradient Descent Optimizer with the input learning rate
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op=optimizer.minimize(loss)
    return train_op

def evaluation(logits, one_hot_label):
    """
    Input
    logits: Logits tensor, float - [batch_size, NUM_CLASSES]
    labels: Labels tensor, int32 - [batch_size] values in the range [0, NUM_CLASSES]

    Output
    A scalar int 32 tensor with the number of examples (out of batch_size)
    """
    prediction=tf.argmax(logits, 1)
    correct_prediction=tf.equal(prediction, tf.argmax(one_hot_label, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def fill_feed_dict(batch_images, batch_labels, input_layer, input_label, batch_size):
    feed={
        input_layer : np.reshape(batch_images, (batch_size,3072)),
        input_label : batch_labels
    }
    return feed

def do_eval(sess,
            eval_correct,
            images,
            labels,
            images_placeholder,
            labels_placeholder,
            numDataInEpoch,
            batch_size
            ):
    """Runs one evaluation against the full epoch of data.
        Input:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    true_count=0
    for start_idx in range(0, len(images), batch_size):
        batch_images=images[start_idx:start_idx+batch_size]
        batch_labels=labels[start_idx:start_idx+batch_size]
        feed=fill_feed_dict(batch_images, batch_labels, images_placeholder, labels_placeholder, batch_size)
        true_count+=sess.run(eval_correct, feed_dict=feed)
    precision=float(true_count)/numDataInEpoch
    print('Num Examples: %d Num Correct:%d Accuracy: %0.5f'%(numDataInEpoch, true_count, precision))

def run_training(input_data):
    """Training CIFAR for a number of steps"""
    print("Initializing Training")

    images_placeholder, labels_placeholder=placeholder_inputs(FLAGS.batch_size)
    label_one_hot=tf.one_hot(labels_placeholder, 10)
    label_one_hot=tf.reshape(label_one_hot, [-1, 10])
    print("labelOneHot", label_one_hot)
    print("label", labels_placeholder)
    logits, output_Layer=inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3)

    lossVal=loss(logits, label_one_hot)

    train_op=training(lossVal, FLAGS.learning_rate)

    summary=tf.summary.merge_all()

    eval_correct=evaluation(logits, label_one_hot)

    init=tf.global_variables_initializer()

    saver=tf.train.Saver()

    sess=tf.Session()

    summary_writer=tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)
    images=input_data.images
    labels=input_data.labels

    import time
    start_time=time.time()
    for epoch in range(FLAGS.max_epoch):
        for start_idx in range(0, len(images), FLAGS.batch_size):
            batch_images=images[start_idx:start_idx+FLAGS.batch_size]
            batch_labels=labels[start_idx:start_idx+FLAGS.batch_size]

            feed=fill_feed_dict(batch_images, batch_labels, images_placeholder, labels_placeholder, FLAGS.batch_size)
            _, loss_value, acc=sess.run([train_op, lossVal, eval_correct], feed_dict=feed)
        duration=time.time()-start_time

        if epoch%2==0:
            print('Step %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, duration))
            summary_str=sess.run(summary, feed_dict=feed)
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()
        if (epoch+1)%10==0 or (epoch+1)==FLAGS.max_epoch:
            checkpoint_file=os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)
            print('Training Data Eval:')
            do_eval(sess, eval_correct, images, labels, images_placeholder, labels_placeholder, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, FLAGS.batch_size)