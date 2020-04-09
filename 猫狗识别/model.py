import tensorflow as tf

def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape,mean=0,stddev=1.0))
    return w

def bias_variables(shape):
    ##为什么要float类型
    b = tf.Variable(tf.constant(0.0,shape=shape))
    return b

def inference(images, batch_size, n_classes):

    ##第一层卷积池化
    with tf.variable_scope("conv1_pool1") as scope:
        w_conv1 = weight_variables([3,3,3,16])
        b_conv1 = bias_variables([16])
        ##卷积激活
        x_relu1 = tf.nn.relu(tf.nn.conv2d(images,w_conv1,strides = [1,1,1,1],padding="SAME")+b_conv1)
        ##池化一波张量形状[-1,104,104,16]
        x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("conv2_pool2"):
        w_conv2 = weight_variables([3,3,16,16])
        b_conv2 = bias_variables([16])
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1,w_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
        ##池化第二波张量形状[-1,52,52,16]
        x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope('full_connect') as scope:
        reshape = tf.reshape(x_pool2, shape=[batch_size, -1])

        ##这一步具体是怎么样的？？？？？？？？？？？？？？？
        dim = reshape.get_shape()[1].value

        weights = weight_variables([dim,2])
        biases = bias_variables([2])
        y_predict =tf.matmul(reshape, weights) + biases

        return y_predict



def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # train_op = optimizer.minimize(loss, global_step=global_step)
        train_op =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        print(correct)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

