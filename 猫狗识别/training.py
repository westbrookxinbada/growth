import tensorflow as tf

import input_data
import model


N_CLASSES = 2
IMG_W = 208  # 重新定义图片的大小，图片如果过大则训练比较慢
IMG_H = 208
BATCH_SIZE = 32  # 每批数据的大小
CAPACITY = 256
MAX_STEP = 10000  # 训练的步数
learning_rate = 0.0001  # 学习率，


def run_training():
    train_dir = './cats_dogs_train/'
    ##存放路径
    logs_train_dir = "./saver/"
    image_list,label_list = input_data.get_files(train_dir)
    """
    获取图片和标签
    """
    image_batch,label_batch = input_data.get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    """
    获得预测值
    """
    logit = model.inference(image_batch,BATCH_SIZE,N_CLASSES)
    """
    sofmax,将其转化为概率
    """
    logit = tf.nn.softmax(logit)

    loss = model.losses(logit,label_batch)
    train_op = model.trainning(loss,learning_rate)
    accuracy = model.evaluation(logit,label_batch)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init_op)
        for i in range(MAX_STEP):
            sess.run(train_op)
            print("训练第%d步，准确率为%f" % (i,
                                     sess.run(accuracy)
                                     ))
            if i==1000:
                saver.save(sess,"./saver/model.ckpt")

        coord.request_stop()
        coord.join(threads)

run_training()