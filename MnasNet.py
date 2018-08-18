import tensorflow as tf

Detection_or_Classifier='classifier'#detection,classifier

class MnasNet():
    def __init__(self,num_classes=1000,learning_rate=0.1):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        self.loss = Loss()
        
        self.__build()
        
    def __build(self):
        self.norm = 'batch_norm'
        self.activate = 'relu'
        self.infos = {#num_filters,kernel_size,ratio,BlockRepeat
                      '1':[24,3,3,[1,2]],
                      '2':[40,5,3,[1,2]],
                      '3':[80,5,6,[1,2]],
                      '4':[96,3,6,[0,1]],
                      '5':[192,5,6,[1,3]],
                      '6':[320,3,6,[0,0]]
                      }
        
        self.__init_global_epoch()
        self.__init_BestScore()
        self.__init_input()
        
        with tf.variable_scope('zsc_feature'):
            x = PrimaryConv('PrimaryConv', self.input_image, self.norm, self.activate, self.is_training)
            
            index = '1'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
            index = '2'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
            index = '3'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
            index = '4'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
            index = '5'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
            index = '6'
            x = MnasNetBlock('MnasNetBlock_{}'.format(index), x, self.infos[index][0], self.infos[index][1], self.infos[index][2], self.infos[index][3],
                             self.norm, self.activate, self.is_training)
            
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('zsc_classifier'):
                x = tf.nn.avg_pool(x, [1,7,7,1], [1,1,1,1], 'VALID')
                x = _conv_block('FinalConv',x,self.num_classes,1,1,'SAME',None,None,self.is_training)
                self.classifier_logits = tf.reshape(x, [-1, self.num_classes])
        elif Detection_or_Classifier=='detection':
            pass
        else:
            raise ValueError('Detection_or_Classifier must be Classifier or Detection')
        
        self.__init__output()
    
    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = self.loss.regularzation_loss()
            
            if Detection_or_Classifier=='classifier':
                self.all_loss = self.loss.sparse_softmax_loss(self.classifier_logits,self.y)
                self.all_loss += regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=1,decay_rate=0.995)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
                
                self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
                self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_out_softmax,self.y,5),tf.float32))

            elif Detection_or_Classifier=='detection':
                pass
    def __init_input(self):
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32, [None,224,224,3], name='zsc_input')
                self.y = tf.placeholder(tf.int32, [None], name='zsc_input_target')
                self.is_training = tf.placeholder(tf.bool, name='zsc_is_training')
        elif Detection_or_Classifier=='detection':
            pass
        else:
            raise ValueError('Detection_or_Classifier must be Detection or Classifier')
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_BestScore(self):
        with tf.variable_scope('BestScore'):
            self.BestScore_tensor = tf.Variable(0.0, trainable=False, name='BestScore_tensor')
            self.BestScore_input = tf.placeholder('float32', None, name='BestScore_input')
            self.BestScore_assign_op = self.BestScore_tensor.assign(self.BestScore_input)

from tensorflow.python.ops import array_ops
##LOSS
class Loss():
    def __init__(self):
        pass
    #regularzation loss
    def regularzation_loss(self):
        return sum(tf.get_collection("regularzation_loss"))
    
    #sparse softmax loss
    def sparse_softmax_loss(self, logits, labels):
        labels = tf.to_int32(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
            logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    #focal loss
    def focal_loss(self, prediction_tensor, target_tensor, alpha=0.25, gamma=2):
        #prediction_tensor [batch,num_anchors,num_classes]
        #target_tensor     [batch,num_anchors,num_classes]
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
        
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent,2)
    
    #smooth_L1
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x),1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))
    
    def ssd_loss(self, num_classes, pred, ground_truth, positive, negative, use_focal_loss=True):
        #pred [batch,num_anchors,num_classes+4]
        #ground_truth [batch,num_anchors,1+4]
        #positive [batch,num_anchors]
        #negative [batch,num_anchors]
        ground_truth_count = tf.add(positive,negative)
        if use_focal_loss:
            loss_class = self.focal_loss(pred[:,:,1:-4],tf.one_hot(tf.cast(ground_truth[:,:,0],tf.int32),num_classes))
        else:
            loss_class = self.sparse_softmax_loss(pred[:,:,1:-4],tf.cast(ground_truth[:,:,0],tf.int32))
        self.loss_location = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                  tf.reduce_sum(
                                                                                self.smooth_L1(
                                                                                               tf.subtract(
                                                                                                           ground_truth[:,:,1:], 
                                                                                                           pred[:,:,-4:]
                                                                                                           )
                                                                                               ),
                                                                                2
                                                                                ), 
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_class = tf.truediv(
                                     tf.reduce_sum(
                                                   tf.multiply(
                                                               loss_class,
                                                               ground_truth_count),
                                                   1), 
                                     tf.reduce_sum(ground_truth_count,1)
                                     )
        self.loss_confidence = tf.truediv(
                                        tf.reduce_sum(
                                                      tf.multiply(
                                                                 self.smooth_L1(
                                                                                tf.subtract(
                                                                                            positive, 
                                                                                            pred[:,:,0]
                                                                                            )
                                                                                ),
                                                                  positive
                                                                  ),
                                                      1), 
                                        tf.reduce_sum(positive,1)
                                        )
        self.loss_unconfidence = tf.truediv(
                                            tf.reduce_sum(
                                                          tf.multiply(
                                                                     self.smooth_L1(
                                                                                    tf.subtract(
                                                                                                negative, 
                                                                                                pred[:,:,0]
                                                                                                )
                                                                                    ),
                                                                      negative
                                                                      ),
                                                          1), 
                                            tf.reduce_sum(ground_truth_count,1)
                                            )
        return self.loss_class,self.loss_location,self.loss_confidence,self.loss_unconfidence
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##################################################LAYERS##################################################################
##MnasNetBlock
def MnasNetBlock(name,x,num_filters=24,kernel_size=3,ratio=3,BlockRepeat=[1,2],norm='batch_norm',activate='relu',is_training=True):
    with tf.variable_scope(name):
        if BlockRepeat[0]==1:
            x = DepthwiseConvBlock('Block_0',x,num_filters,kernel_size,2,ratio,False,norm,activate,is_training)
        elif BlockRepeat[0]==0:
            x = DepthwiseConvBlock('Block_0',x,num_filters,kernel_size,1,ratio,False,norm,activate,is_training)
        else:
            raise ValueError('BlockRepeat[0] must be 0 or 1')
        
        for i in range(BlockRepeat[1]):
            x = DepthwiseConvBlock('Block_{}'.format(i+1),x,num_filters,kernel_size,1,ratio,True,norm,activate,is_training)
        
        return x
##PrimaryConv
def PrimaryConv(name,x,norm='batch_norm',activate='relu',is_training=True):
    with tf.variable_scope(name):
        x = BN(x, is_training, 'PrimaryBN')
        x = _conv_block('conv_0', x, 32, 3, 2, 'SAME', norm, activate, is_training)
        x = _depthwise_conv2d('depthwise_conv2d_0', x, 1, 3, 1, 'SAME', norm, activate, is_training)
        x = _conv_block('conv_1', x, 16, 3, 1, 'SAME', norm, activate, is_training)
        return x
##DepthwiseConvBlock
def DepthwiseConvBlock(name,x,num_filters=24,kernel_size=3,stride=1,ratio=3,isAdd=True,norm='batch_norm',activate='relu',is_training=True):
    with tf.variable_scope(name):
        input = x
        x = _conv_block('unCompress',x,ratio*num_filters,1,1,'SAME',norm,activate,is_training)
        x = _depthwise_conv2d('depthwise_conv2d',x,1,kernel_size,stride,'SAME',norm,activate,is_training)
        x = _conv_block('Compress',x,num_filters,1,1,'SAME',norm,activate,is_training)
        if isAdd:
            x += input
        return x
##__conv_block
def _conv_block(name,x,num_filters=32,kernel_size=3,stride=1,padding='SAME',norm='batch_norm',activate='relu',is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight', [kernel_size, kernel_size, x.shape.as_list()[-1], num_filters])
        x = tf.nn.conv2d(x, w, [1,stride,stride,1], padding=padding, name='conv_block')
        
        b = tf.get_variable('bias', [num_filters], tf.float32, initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b)
        
        _norm = NORM(norm).call()
        _activate = ACTIVATE(activate).call()
        
        if norm:
            x = _norm(x, is_training, name=norm)
        else:
            pass
        if activate:
            x = _activate(x, name=activate)
        else:
            pass
        
        return x
##__depthwise_conv2d
def _depthwise_conv2d(name,x,scale=1,kernel_size=3,stride=1,padding='SAME',norm='batch_norm',activate='relu',is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight', [kernel_size, kernel_size, x.shape.as_list()[-1], scale])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding, name='depthwise_conv2d')
        
        b = tf.get_variable('bias', [scale*x.shape.as_list()[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, b)
        
        _norm = NORM(norm).call()
        _activate = ACTIVATE(activate).call()
        
        if norm:
            x = _norm(x,is_training,name=norm)
        else:
            pass
        if activate:
            x = _activate(x,name=activate)
        else:
            pass
        
        return x

class NORM():
    def __init__(self,name='batch_norm'):
        self.name = name
    def call(self):
        if self.name=='batch_norm':
            return BN
class ACTIVATE():
    def __init__(self,name='relu'):
        self.name = name
    def call(self):
        if self.name=='relu':
            return tf.nn.relu
##batch_norm
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
def BN(x, is_training, name='batch_norm'):
    with tf.variable_scope(name):
        decay = 0.99
        epsilon = 1e-3
        
        size = int(x.shape.as_list()[-1])
        
        beta = tf.get_variable('beta', [size], initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [size], initializer=tf.ones_initializer())

        moving_mean = tf.get_variable('mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('variance', [size], initializer=tf.ones_initializer(), trainable=False)

        def train():
            mean, variance = tf.nn.moments(x, [0,1,2])
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
            return mean, variance

        mean, variance = control_flow_ops.cond(
                                               tf.convert_to_tensor(is_training,dtype=tf.bool), 
                                               lambda: train(),
                                               lambda: (moving_mean, moving_variance)
                                               )
     
        return tf.nn.batch_normalization(x, mean, variance, beta, scale, epsilon)
##weight initializer
def GetWeight(name,shape,weights_decay=0.000004):
    with tf.variable_scope(name):
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        weight_decay = tf.multiply(tf.nn.l2_loss(w),weights_decay,name='weight_loss')
        tf.add_to_collection('regularzation_loss',weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)

class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def _compute_fans(self,shape):
        if len(shape) < 1:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            receptive_field_size = 1.
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return fan_in, fan_out
    
    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = self._compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)

if __name__=='__main__':
    from functools import reduce
    from operator import mul
    
    import numpy as np

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    model = MnasNet(num_classes=1000)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        feed_dict = {
                     model.input_image: np.random.randn(1,224,224,3),
                     model.y: [1],
                     model.is_training: True
                     }
        
        out = sess.run(model.all_loss, feed_dict=feed_dict)
        print(out)