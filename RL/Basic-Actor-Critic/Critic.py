import tensorflow as tf
import numpy as np

class Critic(object):

    def __init__(self,sess,n_features,gamma = 0.9,lr=0.01):

        self.sess = sess
        self.s = tf.placeholder(tf.float32,[1,n_features],name='state')
        self.v_ = tf.placeholder(tf.float32,[1,1],name='v_next')
        self.r = tf.placeholder(tf.float32,None,name='r')

        # 同Actor一样，我们的Critic也是一个双层的神经网络结构
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0,0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'l1'
            )

            self.v = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation = None,
                kernel_initializer=tf.random_normal_initializer(0,0.1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'V'
            )

        """
        Critic要反馈给Actor一个时间差分值，来决定Actor选择动作的好坏，如果时间差分值大的话，说明当前Actor选择的这个动作的惊喜度较高，
        需要更多的出现来使得时间差分值减小。
        
        考虑时间差分的计算：
        TD = r + gamma * f(s') - f(s),这里f(s)代表将s状态输入到Critic神经网络中得到的Q值。
        所以Critic的输入也分三个，首先是当前状态，当前的奖励，以及下一个时刻的奖励折现值。为什么没有动作A呢？动作A是确定的呀，是Actor选的呀，
        对不对！还有为什么不是下一时刻的Q值而不是下一个时刻的状态，因为我们已经在计算TD时已经把状态带入到神经网络中得到Q值了。
        """
        # Critic的损失定义为时间差分值的平方值
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)


        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    # Critic的任务就是告诉Actor当前选择的动作好不好，所以我们只要训练得到TD并返回给Actor就好
    def learn(self,s,r,s_):
        s,s_ = s[np.newaxis,:],s_[np.newaxis,:]

        v_ = self.sess.run(self.v,feed_dict = {self.s:s_})

        td_error,_ = self.sess.run([self.td_error,self.train_op],
                                   feed_dict={self.s:s,self.v_:v_,self.r:r})

        return td_error



