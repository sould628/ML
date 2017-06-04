import tensorflow as tf
import numpy as np
import code.CIFAR_INPUT as CIFAR_INPUT
import os

IMAGE_PIXELS=CIFAR_INPUT.IMAGE_PIXELS
NUM_CLASSES=CIFAR_INPUT.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=CIFAR_INPUT.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

tf.app.flags.DEFINE_integer('batch_sizeM', 100, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('total_epochsM', 50, """Number of Total Epochs""")
tf.app.flags.DEFINE_float('learning_rateM', 0.001, """Learning Rate""")
tf.app.flags.DEFINE_string('log_dirM', './log/','''Path to the log''')
FLAGS=tf.app.flags.FLAGS

def inference(images):
    """Build CIFAR-10 Model"""
    """Input
        images: Image_Placeholder
        hiddenX_units: size of the X-th hidden layer
        Output: Logits"""
    #hidden1

    with tf.name_scope('conv1'):
        #conv+bias

        kernel1 = tf.get_variable('kernel1', shape=[5, 5, 3, 64])
        c1=tf.nn.conv2d(images, kernel1, [1,1,1,1], padding='SAME')
        b1=tf.get_variable('bias1', shape=[64], initializer=tf.zeros_initializer())
        bias1=tf.nn.bias_add(c1, b1)

        #activation
        conv1=tf.nn.relu(bias1)
        print(conv1)
        #pool
        pool1=tf.nn.max_pool(conv1, [1,3,3,1], strides=[1,2,2,1], padding='SAME')
        print(pool1)

    with tf.name_scope('conv2'):
        #conv+bias
        kernel2 = tf.get_variable('kernel2', shape=[5, 5, 64, 64])
        c2=tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding='SAME')
        b2=tf.get_variable('bias2', shape=[64], initializer=tf.zeros_initializer())
        bias2=tf.nn.bias_add(c2, b2)

        #activation
        conv2=tf.nn.relu(bias2)
        print(conv2)
        #pool
        pool2=tf.nn.max_pool(conv2, [1,3,3,1], strides=[1,2,2,1], padding='SAME')
        print(pool2)

    with tf.name_scope('hidden3'):
        dim=1
        for d in pool2.get_shape()[1:].as_list():
            dim*=d
        w3=tf.get_variable(
            'w3',
            shape=[dim, 384],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        pool2=tf.reshape(pool2, [-1, w3.get_shape().as_list()[0]])
        b3=tf.get_variable('b3', shape=[384], initializer=tf.zeros_initializer())
        hidden3=tf.nn.relu(tf.matmul(pool2, w3)+b3)
        print(hidden3)

    with tf.name_scope('hidden4'):
        w4=tf.get_variable(
            'w4',
            shape=[384, 192],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b4=tf.get_variable('b4', shape=[192], initializer=tf.zeros_initializer())
        hidden4=tf.nn.relu(tf.matmul(hidden3, w4)+b4)
        print(hidden4)

    with tf.name_scope('softmax_linear'):
        wo=tf.get_variable(
            'wo',
            shape=[192, NUM_CLASSES],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bo=tf.get_variable('bo', shape=[NUM_CLASSES], initializer=tf.zeros_initializer())
        logits=tf.matmul(hidden4, wo) + bo
        print(logits)
    return logits

def loss(logits, label_one_hot):
    """LOSS VALUE"""
    lossval=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
    return lossval

def training(lossVal, learning_rate):
    """TRAINING BY GRADIENTDESCENTMETHODS"""
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op=optimizer.minimize(lossVal)
    return train_op

def evaluation(logits, label_one_hot):
    _y=tf.nn.softmax(logits)
    learnedValue=tf.argmax(_y, 1)
    correctValue=tf.equal(learnedValue, tf.argmax(label_one_hot, 1))
    return tf.reduce_mean(tf.cast(correctValue, tf.float32))

def trainConvMyModel(data):
    tf.reset_default_graph()
    import time
    print ("Training with MyOwn Conv model")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #inputs
    images_placeholder = tf.placeholder(tf.float32, shape = [None, 32,32,3])
    labels_placeholder = tf.placeholder(tf.int32, shape = [None, 1])
    label_one_hot = tf.one_hot(labels_placeholder, 10)
    label_one_hot = tf.reshape(label_one_hot, [-1, 10])


    logits = inference(images_placeholder)
    lossVal = loss(logits, label_one_hot)
    train_op = training(lossVal, learning_rate=FLAGS.learning_rateM)
    accuracy = evaluation(logits, label_one_hot)

    summary=tf.summary.merge_all()
    saver=tf.train.Saver()
    summary_writer=tf.summary.FileWriter(FLAGS.log_dirM, sess.graph)

    images=data.data_images
    labels=data.data_labels
    test_images=data.test_images
    test_labels=data.test_labels
    start_time=time.time()
    sess.run(tf.global_variables_initializer())
    loss_history=[]
    for epoch in range(FLAGS.total_epochsM):
        for start_idx in range(0, len(images), FLAGS.batch_sizeM):
            batch_images = images[start_idx:start_idx+FLAGS.batch_sizeM]
            batch_labels = labels[start_idx:start_idx+FLAGS.batch_sizeM]

            feed = {images_placeholder : np.reshape(batch_images, (-1, 32, 32, 3)),
                    labels_placeholder : np.reshape(batch_labels, (-1, 1))}
            _, loss_value, acc= sess.run([train_op, lossVal, accuracy], feed_dict = feed)
            loss_history.append(loss_value)
        duration=time.time()-start_time
        print('epoch %d: loss = %.5f acc=%.2f (%.5f sec)' % (epoch, loss_value, acc, duration))

        if (epoch+1)%10==0 or (epoch+1)==FLAGS.total_epochsM:
            checkpoint_file=os.path.join(FLAGS.log_dirM, 'MConvmodel.ckpt')
            saver.save(sess, checkpoint_file, global_step=epoch)
            print('TEST Data Eval(MyOwn): ')
            true_count=0
            for start_idx in range(0, len(test_images), FLAGS.batch_sizeM):
                batch_images=test_images[start_idx:start_idx+FLAGS.batch_sizeM]
                batch_labels=test_labels[start_idx:start_idx+FLAGS.batch_sizeM]
                test_feed = {images_placeholder : np.reshape(batch_images, (-1, 32, 32, 3)),
                             labels_placeholder : np.reshape(batch_labels, (-1, 1))}
                true_count+=(FLAGS.batch_sizeM*sess.run(accuracy, feed_dict=test_feed))
            precision=float(true_count)/NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
            print('Num Examples: %d Num Correct:%d Accuracy: %0.5f'%(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, true_count, precision))
    print("Result of MyOwn Conv Model with Test Set: ", precision)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title('loss history (MyOwn Conv Model)')
    plt.plot(loss_history)
    plt.show()


