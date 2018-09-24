import tensorflow as tf
import model_helper

# 深层自编码类
class DAE:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.weight_initializer = model_helper._get_weight_initializer()
        self.bias_initializer = model_helper._get_bias_initializer()
        self.init_parameters()

    def init_parameters(self):
        '''Initialize networks weights abd biasis.'''
        
        with tf.name_scope('weights'):
            self.W_1=tf.get_variable(name='weight_1', shape=(self.FLAGS.num_v, self.FLAGS.num_h),
                                     initializer=self.weight_initializer)
            self.W_2=tf.get_variable(name='weight_2', shape=(self.FLAGS.num_h, self.FLAGS.num_h),
                                     initializer=self.weight_initializer)
            self.W_3=tf.get_variable(name='weight_3', shape=(self.FLAGS.num_h, self.FLAGS.num_h),
                                     initializer=self.weight_initializer)
            self.W_4=tf.get_variable(name='weight_4', shape=(self.FLAGS.num_h, self.FLAGS.num_v),
                                     initializer=self.weight_initializer)
        
        with tf.name_scope('biases'):
            self.b1=tf.get_variable(name='bias_1', shape=(self.FLAGS.num_h), 
                                    initializer=self.bias_initializer)
            self.b2=tf.get_variable(name='bias_2', shape=(self.FLAGS.num_h), 
                                    initializer=self.bias_initializer)
            self.b3=tf.get_variable(name='bias_3', shape=(self.FLAGS.num_h), 
                                    initializer=self.bias_initializer)
        
    def _inference(self, x):
        '''
        @param x: 输入 用户-电影 矩阵
        @return : networks predictions
        '''
        with tf.name_scope('inference'):
             a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
             a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
             a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))   
             a4=tf.matmul(a3, self.W_4) 
        return a4
    
    def _compute_loss(self, predictions, labels, num_labels):
        '''
        @param predictions: predictions of the stacked autoencoder//堆叠自编码器
    	@param labels: input values of the stacked autoencoder which serve as labels at the same time
    	@param num_labels: number of labels !=0 in the data set to compute the mean
    	@return mean squared error loss tf-operation
    	'''
        with tf.name_scope('loss'):
            # 均方差：差+平方+平均
            loss_op=tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions, labels))), num_labels)
            return loss_op

    # ====================
    def _optimizer(self, x):
        '''
        @param x: input values for the stacked autoencoder
        @return: tensorflow training operation
        @return: root mean squared error
        '''
        outputs = self._inference(x)
        mask = tf.where(tf.equal(x, 0.0), tf.zeros_like(x), x) # indices of 0 values in the training set
        num_train_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float32) # 标量，单个样本中非0的数据个数
        bool_mask = tf.cast(mask, dtype=tf.bool)
        outputs = tf.where(bool_mask, outputs, tf.zeros_like(outputs))

        MSE_loss = self._compute_loss(outputs, x, num_train_labels)
        
        if self.FLAGS.l2_reg==True:
            # tf.add_n 将拥有相同维度的张量相加
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            MSE_loss = MSE_loss + self.FLAGS.lambda_ * l2_loss
        
        train_op=tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(MSE_loss)
        RMSE_loss=tf.sqrt(MSE_loss)

        return train_op, RMSE_loss
    # ====================
    def _validation_loss(self, x_train, x_test):
        ''' Computing the loss during the validation time.
    	  @param x_train: training data samples
    	  @param x_test: test data samples
    	  @return networks predictions
    	  @return root mean squared error loss between the predicted and actual ratings
    	  '''
        outputs = self._inference(x_train)
        mask = tf.where(tf.equal(x_test, 0.0), tf.zeros_like(x_test), x_test)
        num_test_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float32)
        bool_mask = tf.cast(mask, dtype=tf.bool)
        outputs = tf.where(bool_mask, outputs, tf.zeros_like(outputs))
    
        MSE_loss = self._compute_loss(outputs, x_test, num_test_labels)
        RMSE_loss = tf.sqrt(MSE_loss)

        return outputs, RMSE_loss