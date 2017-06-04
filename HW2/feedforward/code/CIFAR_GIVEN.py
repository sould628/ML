import tensorflow as tf
import numpy as np
import code.CIFAR_INPUT as CIFAR_INPUT
import os

IMAGE_PIXELS=CIFAR_INPUT.IMAGE_PIXELS
NUM_CLASSES=CIFAR_INPUT.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

tf.app.flags.DEFINE_integer('batch_sizeG', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('total_epochsG', 100, """Number of Total Epochs""")
tf.app.flags.DEFINE_integer('hidden1G', 1024, """Number of 1st hidden units""")
tf.app.flags.DEFINE_integer('hidden2G', 384, """Number of 2nd hidden units""")
tf.app.flags.DEFINE_integer('hidden3G', 192, """Number of 3rd hidden units""")
tf.app.flags.DEFINE_float('learning_rateG', 0.1, """Learning Rate""")
tf.app.flags.DEFINE_string('log_dirG', './log/','''Path to the log''')
FLAGS=tf.app.flags.FLAGS

def inference(images, hidden1_units, hidden2_units, hidden3_units):
    """Build CIFAR-10 Model"""
    """Input
        images: Image_Placeholder
        hiddenX_units: size of the X-th hidden layer
        Output: Logits"""
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
    return output_layer

def loss(output_layer, label_one_hot):
    """LOSS VALUE"""
    return tf.reduce_mean(-tf.reduce_sum(label_one_hot*tf.log(output_layer), axis=1))

def training(lossVal, learning_rate):
    """TRAINING BY GRADIENTDESCENTMETHODS"""
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op=optimizer.minimize(lossVal)
    return train_op

def evaluation(output_layer, label_one_hot):
    learnedValue=tf.argmax(output_layer, 1)
    correctValue=tf.equal(learnedValue, tf.argmax(label_one_hot, 1))
    return tf.reduce_mean(tf.cast(correctValue, tf.float32))

def trainFFGivenModel(data):
    tf.reset_default_graph()
    import time
    print ("Training with Given FF model")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #inputs
    images_placeholder = tf.placeholder(tf.float32, shape = [None, 3072])
    labels_placeholder = tf.placeholder(tf.int32, shape = [None, 1])
    label_one_hot = tf.one_hot(labels_placeholder, 10)
    label_one_hot = tf.reshape(label_one_hot, [-1, 10])


    output_layer = inference(images_placeholder, FLAGS.hidden1G, FLAGS.hidden2G, FLAGS.hidden3G)
    lossVal = loss(output_layer, label_one_hot)
    train_op = training(lossVal, learning_rate=FLAGS.learning_rateG)
    accuracy = evaluation(output_layer, label_one_hot)

    summary=tf.summary.merge_all()
    saver=tf.train.Saver()
    summary_writer=tf.summary.FileWriter(FLAGS.log_dirG, sess.graph)

    images=data.data_images
    labels=data.data_labels
    start_time=time.time()
    sess.run(tf.global_variables_initializer())
    loss_history=[]
    for epoch in range(FLAGS.total_epochsG):
        for start_idx in range(0, len(images), FLAGS.batch_sizeG):
            batch_images = images[start_idx:start_idx+FLAGS.batch_sizeG]
            batch_labels = labels[start_idx:start_idx+FLAGS.batch_sizeG]

            feed = {images_placeholder : np.reshape(batch_images, (-1, IMAGE_PIXELS)),
                    labels_placeholder : np.reshape(batch_labels, (-1, 1))}
            _, loss_value, acc = sess.run([train_op, lossVal, accuracy], feed_dict = feed)
            loss_history.append(loss_value)

        duration=time.time()-start_time
        print('epoch %d: loss = %.5f (%.5f sec)' % (epoch, loss_value, duration))

        if (epoch+1)%10==0 or (epoch+1)==FLAGS.total_epochsG:
            checkpoint_file=os.path.join(FLAGS.log_dirG, 'GFFmodel.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)
            print('Training Data Eval(Given): ')
            do_eval(sess, accuracy, images, labels, images_placeholder, labels_placeholder, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, FLAGS.batch_sizeG)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title('loss history (Given FF Model)')
    plt.plot(loss_history)
    plt.show()

    test_images=data.test_images
    test_labels=data.test_labels
    test_feed={
        images_placeholder:np.reshape(test_images, (-1, IMAGE_PIXELS)),
        labels_placeholder:np.reshape(test_labels, (-1, 1))
        }
    test_acc=sess.run([accuracy], feed_dict=test_feed)
    print("Result of Given FF Model with Test Set: ", test_acc)

def do_eval(sess,
            accuracy,
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
        feed = {images_placeholder : np.reshape(batch_images, (-1, IMAGE_PIXELS)),
                labels_placeholder : np.reshape(batch_labels, (-1, 1))}
        true_count+=(batch_size*sess.run(accuracy, feed_dict=feed))
    precision=float(true_count)/numDataInEpoch
    print('Num Examples: %d Num Correct:%d Accuracy: %0.5f'%(numDataInEpoch, true_count, precision))