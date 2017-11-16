import tensorflow as tf
import numpy as np
import string


def variable_summaries(var):
    vname = string.join(var.name.split(":"), "")
    
    with tf.name_scope(vname):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def softplus(x):
    return tf.log(1 + tf.exp(x))

def normpdf(x, m, r):
    return (1 / tf.sqrt(2 * np.pi * (r**2))) * tf.exp(-(x - m)**2/2/r**2)

def fsoftmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), 2, keep_dims = True)

def dummyfunc(x):
    return x

class bcnn_conv(object):
    
    def __init__(self, model, inp, shape, w_stdev, n_samples, pooling = True, outact = tf.sigmoid, layernum = 0):
        with tf.name_scope('conv_layer' + str(layernum)):
            #shape[0] += 1
            
            self.w = tf.Variable(tf.truncated_normal(shape, stddev = w_stdev), dtype = tf.float32, name = 'weight')
            self.n_samples = n_samples
            variable_summaries(self.w)

            self.conv = tf.nn.conv2d(inp, self.w, strides = [1, 1, 1, 1], padding = 'SAME')
            self.dropout = tf.nn.dropout(self.conv, model.keep_probs[layernum])
            self.pre_o = outact(self.dropout)
            
            if pooling:
                self.pool = tf.nn.max_pool(self.pre_o, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
                self.out = self.pool
            else:
                self.out = self.pre_o
            
            tf.summary.histogram('activation', self.out)

            self.params = [self.w]

            print ("conv layer done")


class bcnn_fc(object):
    
    def __init__(self, model, inp, shape, w_stdev, n_samples, outact = tf.sigmoid, layernum = 0):
        with tf.name_scope('fc_layer' + str(layernum)):
            shape[0] += 1
            
            self.w = tf.Variable(tf.truncated_normal(shape, stddev = w_stdev), dtype = tf.float32, name = 'weight')
            #self.b = tf.Variable(tf.constant(w_stdev, shape = shape), dtype = tf.float32, name = 'bias')
            self.n_samples = n_samples
            variable_summaries(self.w)
            
            shape_inp = tf.shape(inp)
            iones = tf.ones(shape = [shape_inp[0], 1])
            newinp = tf.concat(axis = 1, values = [inp, iones])

            self.pre_o = tf.matmul(newinp, self.w)
            self.dropout = tf.nn.dropout(self.pre_o, model.keep_probs[layernum])
            self.out = outact(self.dropout)
            tf.summary.histogram('activation', self.out)

            self.params = [self.w]

            print ("fc layer done")

class bcnn_gal_model(object):
    
    def __init__(self, shape_img, shape_conv, pool_conv, shape_fc, w_stdev = 0.1, n_samples = 10, outact = tf.nn.relu, seed = 1234, lr = 1e-8, l2_reg = False, l2_lambda = 0.1):
        
        self.n_layers_conv = len(shape_conv)
        self.n_layers_fc = len(shape_fc) - 1
        #self.n_samples = n_samples
               
        self.x = tf.placeholder(tf.float32, [None] + shape_img, name = 'x')
        self.t = tf.placeholder(tf.float32, [None, shape_fc[-1]], name = 't')
        self.keep_probs = tf.placeholder(tf.float32, [self.n_layers_conv + self.n_layers_fc], name='keep_probs')
        self.n_samples = tf.placeholder(tf.int32, [1], name='n_samples')
        
        #self.x3 = tf.tile(tf.expand_dims(self.x, 0), [self.n_samples, 1, 1])
        self.x3 = tf.tile(self.x, [10, 1, 1, 1])
        
        self.layers = list()
        
        for i in range(self.n_layers_conv):
            if i == 0:
                inp = self.x
                #inp = self.x3
            else:
                inp = self.layers[i-1].out
                
            self.layers.append(bcnn_conv(self, inp, shape_conv[i], w_stdev, pooling = pool_conv[i], n_samples = self.n_samples, outact = outact, layernum = i))
        
        for i in range(self.n_layers_fc):
            if i == 0:
                inp = tf.reshape(self.layers[self.n_layers_conv + i - 1].out, [-1, shape_fc[0]])
            else:
                inp = self.layers[self.n_layers_conv + i - 1].out
            
            if i == self.n_layers_fc-1:
                actout = dummyfunc
            else: 
                actout = outact
                
            self.layers.append(bcnn_fc(self, inp, [shape_fc[i], shape_fc[i+1]], w_stdev, n_samples = self.n_samples, outact = actout, layernum = self.n_layers_conv + i))
        
        #self.shape_out = tf.shape(self.layers[-1].out)
        #self.pred = tf.reshape(self.layers[-1].out, [self.n_samples, -1, shape_fc[-1]])
        self.pred = self.layers[-1].out
        print self.pred.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        with tf.name_scope('terminal'):
            #self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = tf.tile(self.t, [self.n_samples, 1])), name = 'loglike')
            self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred, labels = self.t), name = 'loglike')
            
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.t, 1))), name = 'accuracy')
            #self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.reduce_mean(tf.reshape(self.pred, [self.n_samples, -1, shape_fc[-1]]), [0]), 1), tf.argmax(self.t, 1))), name = 'accuracy')
            
            tf.summary.scalar('loglike', self.loglike)
            tf.summary.scalar('accuracy', self.acc)
            
        #self.n_batches = tf.cast(size_data / size_batch, tf.float32).eval()
        
        if l2_reg:
            self.loss = self.loglike
            for layer in self.layers:
                self.loss += l2_lambda * tf.nn.l2_loss(layer.w)
        else:            
            self.loss = self.loglike
    
        self.params = [p for layer in self.layers for p in layer.params]

        self.init_lr = lr
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(tf.constant(lr), dtype = tf.float32, trainable = False, name='lr')
            tf.summary.scalar('learning_rate', self.learning_rate)
            
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_loss(self):
        return self.loss
        
    def get_params(self):
        return self.params
    
    def print_params(self):
        for layer in self.layers:
            print "weight: {}".format(tf.reduce_mean(layer.params[0]).eval())
    
    def get_inputs(self):
        return [self.x, self.t]
    
    def train_grads(self):
        return self.train_grad
    
    def train(self, feed):
        self.train_op.run(feed_dict = feed)
        
    def validate(self, inp, tar, keep_probs, n_samples):
        
        newfeed = {self.x:np.tile(inp, (n_samples, 1, 1, 1)), self.t: tar, self.keep_probs: keep_probs}
        shape_tar = tf.shape(tar)
        ret = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.reduce_mean(tf.reshape(self.pred, [n_samples, -1, shape_tar[-1]]), [0]), 1), tf.argmax(self.t, 1))), name = 'MCdropout')
        
        return ret.eval(feed_dict = newfeed)
    
    def MCdropout(self, inp, tar, keep_probs, n_samples):
        
        newfeed = {self.x: inp, self.t: tar, self.keep_probs: keep_probs}
        rets = list()
        for i in range(n_samples):
            rets.append(self.pred.eval(feed_dict = newfeed))
        #print np.array(rets).shape
        rets_mean = np.mean(np.array(rets), axis = 0)
        #print rets_mean.shape
        
        newacc = np.mean(np.equal(np.argmax(rets_mean, axis = 1), np.argmax(tar, axis = 1)))
        
        return newacc
        
    def decay_lr(self, rate_decay = 0.5, limit_decay = 1e-8):
        if self.learning_rate.eval() > limit_decay:
            self.learning_rate.assign(self.learning_rate.eval() * rate_decay).eval()
            
    def reset_lr(self):
        self.learning_rate.assign(self.init_lr).eval()
    
    def get_lr(self):
        return self.learning_rate.eval()
    
        
    