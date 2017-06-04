import numpy as np

#GLOBAL VARIABLES
IMAGE_SIZE=32
NUM_CLASSES=10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=10000
IMAGE_PIXELS=IMAGE_SIZE*IMAGE_SIZE*3

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict=pickle.load(fo, encoding='bytes')
        return dict

def load_batch(file):
    raw_data=unpickle(file)
    images=np.reshape(raw_data[b'data'], (-1,32,32,3))
    labels=np.asarray(raw_data[b'labels'])
    return images, labels

def CIFAR2PNG(img, width, height, depth):
    import numpy as np
    color=[0, 32*32, 32*32*2]
    imgR=img[color[0]:color[1]]; imgG=img[color[1]:color[2]]; imgB=img[color[2]:];
    result=[]
    for x in range(0, width):
        col=[]
        for y in range(0, height):
            pixelCoord=x*height+y
            pixel=[imgR[pixelCoord], imgG[pixelCoord], imgB[pixelCoord]]
            col.append(pixel)
        result.append(col)
    return result

def showRandomData(record, labelList):
    import random
    import matplotlib.pyplot as plt
    rBatch=random.randrange(0,5)
    rData1=random.randrange(0,10000)
    rData2=random.randrange(0,10000)
    while(rData2 == rData1):
        rData2=random.randrange(1,10001)

    img1=record.batchfile[rBatch][b'data'][rData1]
    img1_label=record.batchfile[rBatch][b'labels'][rData1]
    img1_name=record.batchfile[rBatch][b'filenames'][rData1]
    img2=record.batchfile[rBatch][b'data'][rData2]
    img2_label=record.batchfile[rBatch][b'labels'][rData2]
    img2_name=record.batchfile[rBatch][b'filenames'][rData2]

    img1_png=CIFAR2PNG(img1, record.width, record.height, record.depth)
    img2_png=CIFAR2PNG(img2, record.width, record.height, record.depth)
    plt.figure(1)
    plt.subplot(121)
    plt.title(img1_name)
    plt.xlabel(labelList[b'label_names'][img1_label])
    plt.imshow(img1_png)

    plt.subplot(122)
    plt.title(img2_name)
    plt.xlabel(labelList[b'label_names'][img2_label])
    plt.imshow(img2_png)
    plt.show()



def read_CIFAR10(filename_queue, testfile, labelfile):

    class CIFAR10Record(object):
        batchfile=[None]*5
    result=CIFAR10Record()
    result.height=32
    result.width=32
    result.depth=3
    result.images=[]
    result.labels=[]
    img_list=[]
    lab_list=[]
    test_img_list=[]
    test_lab_list=[]
    for idx, file in enumerate(filename_queue):
        result.batchfile[idx]=unpickle(file)
        img, lab=load_batch(file)
        img_list.append(img)
        lab_list.append(lab)
    result.data_images=np.vstack(img_list)
    result.data_images=result.data_images.transpose([0,2,1,3])
    result.data_labels=np.hstack(lab_list)
    test_img, test_lab=load_batch(testfile)
    test_img_list.append(test_img)
    test_lab_list.append(test_lab)
    result.test_images=np.vstack(test_img_list)
    result.test_images=result.test_images.transpose([0,2,1,3])
    result.test_labels=np.hstack(test_lab_list)

    labelList=unpickle(labelfile)
#    showRandomData(result, labelList)
    import itertools
    for idx in range(0, 5):
        result.images=result.images+result.batchfile[idx][b'data'].tolist()
        result.labels=result.labels+result.batchfile[idx][b'labels']

    return result
