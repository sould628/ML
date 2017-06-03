import tensorflow as tf

from code import CIFAR_INPUT

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './../file/', """Path to the CIFAR-10 data directory.""")

#GLOBAL CONSTANTS for CIFAR-10 data set.
IMAGE_SIZE=CIFAR_INPUT.IMAGE_SIZE
IMAGE_PIXELS=CIFAR_INPUT.IMAGE_PIXELS
NUM_CLASSES=CIFAR_INPUT.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#Constants for Training Process
MOVING_AVERAGE_DECAY=0.9999
NUM_EPOCHS_PER_DECAY=350.0
LEARNING_RATE_DECAY_FACTOR=0.1
INITIAL_LEARNING_RATE=0.1



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
        weights=tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases=tf.Variables(tf.zeros([hidden1_units]),
                            name='biases')
        hidden1=tf.nn.tanh(tf.matmul(images, weights)+biases)
    #hidden2
    with tf.name_scope('hidden2'):
        weights=tf.Variables(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0/math.sqrt(float(hidden1_units))),
            name='weights')
        biases=tf.Variables(tf.zeros([hidden2_units]),
                            name='biases')
        hidden2=tf.nn.tanh(tf.matmul(hidden1, weights)+biases)
    #hidden3
    with tf.name_scope('hidden3'):
        weights=tf.Variables(
            tf.truncated_normal([hidden2_units, hidden3_units],
                                stddev=1.0/math.sqrt(float(hidden2_units))),
            name='weights')
        biases=tf.Variables(tf.zeros([hidden3_units]),
                            name='biases')
        hidden3=tf.nn.tanh(tf.matmul(hidden2, weights)+biases)

    with tf.name_scope('softmax_linear'):
        weights=tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0/math.sqrt(float(hidden3_units))),
            name='weights')
        biases=tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
        logits=tf.matmul(hidden2, weights) + biases
    return logits

def loss(logits, labels):
    """Calculate the loss from the logits and the labels"""
    """
    Input
    logits: Logits tensor, float ([batch_size, NUM_CLASSES])
    labels: Labels tensor, int32 ([batch_size]
    
    output
    loss: Loss tensor of type float
    """
    labels=tf.to_int64(labels)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy')
    return tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

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
    tf.summary.scalar('loss, loss')
    #Gradient Descent Optimizer with the input learning rate
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #Create a variable to track the global step
    global_step=tf.Variable(0, name='global_step', trainable=False)
    #Use the optimizer to apply the gradients that minimizes the loss and increment the global step counter
    train_op=optimizer.minimize(loss, global_step=global_step)
    return train_op
