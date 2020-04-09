import tensorflow as tf

import input_data
import model

images_list,label_list = input_data.get_files("./cats_dogs_train")
image,label = input_data.get_batch(images_list,label_list,208,208,1,1)


y_predict = model.inference(image,1,2)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess,"./saver")
    print(sess.run(y_predict))