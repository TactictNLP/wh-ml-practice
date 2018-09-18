import tensorflow as tf
import numpy as np


class Actor(object):
    def __init__(self,sess,n_features,n_actions,lr=0.001):
        self.sess = sess

        """
        在这里，由于我们的Actor可以进行单次训练，所以我们的输入只需要是[一个状态]，[一个动作]和[一个奖励]
        """
        self.s = tf.placeholder(tf.float32,[1,n_features],name='state')
        self.a = tf.placeholder(tf.int32,None,name='act')
        self.td_error = tf.placeholder(tf.float32,None,"td_error")

        """
        Actor的神经网络结构和我们的Policy Gradient定义的是一样的，是一个双层的全链接神经网络
        """
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,  # number of hidden units
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.1),  # weights
                bias_initializer = tf.constant_initializer(0.1),  # biases
                name = 'l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs = l1,
                units = n_actions,  # output units
                activation = tf.nn.softmax,  # get action probabilities
                kernel_initializer = tf.random_normal_initializer(mean=0,stddev=0.1),  # weights
                bias_initializer = tf.constant_initializer(0.1),  # biases
                name = 'acts_prob'
            )

            """
            损失函数还是使用的Policy Gradient中提到过的loss= -log(prob)*vt,
            只不过这里的vt换成了由Critic计算出的时间差分误差td_error
            """
            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0,self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            # minimize(-exp_v) = maximize(exp_v)
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    # Actor的训练只需要将状态，动作以及时间差分值喂给网络就可以
    def learn(self,s,a,td):
        s = s[np.newaxis,:]
        feed_dict = {self.s:s,self.a:a,self.td_error:td}
        _,exp_v = self.sess.run([self.train_op,self.exp_v],feed_dict=feed_dict)
        return exp_v

    # 选择动作和Policy Gradient一样，根据计算出的softmax值来选择动作
    def choose_action(self,s):
        s = s[np.newaxis,:]
        probs = self.sess.run(self.acts_prob, feed_dict={self.s:s})  # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]),p=probs.ravel())  # return a int



