import tensorflow as tf
import os
import numpy as np

cats = []
cats_label = []
dogs = []
dogs_label = []
file_dir = "./cats_dogs_train/"
##0为狗，1为猫
def get_files(file_dir):
    files_name = os.listdir(file_dir)
    for file in files_name:
        if "cat" in file:
            cats.append(file_dir+file)
            cats_label.append(1)
        else:
            dogs.append(file_dir+file)
            dogs_label.append(0)
    image_list = np.hstack([cats,dogs])
    label_list = np.hstack([cats_label,dogs_label])
    ##变成一个二维数组第一行是猫狗的文件名，第二行对应的是它的目标值
    # [['./cats_dogs_train/cat.0.jpg' './cats_dogs_train/cat.1.jpg'
    #   './cats_dogs_train/cat.10.jpg'..., './cats_dogs_train/dog.9997.jpg'
    #                                      './cats_dogs_train/dog.9998.jpg' './cats_dogs_train/dog.9999.jpg']
    #  ['1' '1' '1'..., '0' '0' '0']]
    temp = np.array([image_list, label_list])
    ##这里是转置结果如下：
    # [['./cats_dogs_train/cat.0.jpg' '1']
    #  ['./cats_dogs_train/cat.1.jpg' '1']
    #  ['./cats_dogs_train/cat.10.jpg' '1']
    #      ...,
    #  ['./cats_dogs_train/dog.9997.jpg' '0']
    #  ['./cats_dogs_train/dog.9998.jpg' '0']
    #  ['./cats_dogs_train/dog.9999.jpg' '0']]
    temp = np.transpose(temp)
    ##打乱顺序
    np.random.shuffle(temp)
    image_list = (temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]

    return image_list,label_list

def get_batch(image,label,width,height,batch_size,capacity):
    ##转换成tensorflow里的数据格式
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])
    image = input_queue[0]
    label = input_queue[1]

    ##读取文件
    image = tf.read_file(image)
    ##解码
    image = tf.image.decode_jpeg(image,channels=3)
    ##其他方法到时候可以试试入tf.image.resize()
    image = tf.image.resize_image_with_crop_or_pad(image,height,width)
    ##此处转化成的image_batch为Tensor("batch:0", shape=(10, 200, 200, 3), dtype=uint8)
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=80,capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])

    ##转化之后Tensor("Cast_2:0", shape=(10, 200, 200, 3), dtype=float32)
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch,label_batch





