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

class nn_layer(object):
    
    def __init__(self, inp, shape, sb, mu, outact = tf.sigmoid, layernum = 0):
        with tf.name_scope('layer' + str(layernum)):
            shape[0] += 1
            with tf.name_scope('q_pos'):
                self.w = tf.Variable(tf.truncated_normal(shape, stddev = mu), dtype = tf.float32, name = 'mu')
                variable_summaries(self.w)

            with tf.name_scope('p_pri'):
                self.p_w = tf.Variable(tf.constant(0.0, shape = shape), trainable = False, dtype = tf.float32, name = 'p_mu')
                variable_summaries(self.p_w)

            shape_inp = tf.shape(inp)
            iones = tf.ones(shape = [shape_inp[0], 1])
            newinp = tf.concat(1, [inp, iones])

            self.pre_o = tf.batch_matmul(newinp, self.w)
            self.out = outact(self.pre_o)
            tf.summary.histogram('activation', self.out)

            self.params = [self.w]
            self.p_params = [self.p_w]

            print ("layer done")

class nn_model(object):
    
    def __init__(self, shape, size_data, size_batch, mu = 0.1, outact = tf.sigmoid, seed = 1234, lr = 1e-8, only_loglike = False, ewc = False):
        
        self.n_layers = len(shape) - 1
        #self.n_samples = n_samples
               
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name = 'x')
        self.t = tf.placeholder(tf.float32, [None, shape[-1]], name = 't')
        
        #self.x3 = tf.tile(tf.expand_dims(self.x, 0), [self.n_samples, 1, 1])
        
        self.layers = list()
        
        for i in range(self.n_layers):
            if i == 0:
                inp = self.x
            else:
                inp = self.layers[i-1].out
                
            if i == self.n_layers-1:
                actout = dummyfunc
            else: 
                actout = outact
                
            self.layers.append(nn_layer(inp, [shape[i], shape[i+1]], size_batch, mu, outact = actout, layernum = i))
            
        self.pred = self.layers[-1].out
        print self.pred.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        
        with tf.name_scope('terminal'):
            self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(self.pred, [-1, shape[-1]]), self.t), name = 'loglike')

            self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.t, 1))), name = 'accuracy')

            tf.summary.scalar('loglike', self.loglike)
            tf.summary.scalar('accuracy', self.acc)
            
        self.n_batches = tf.cast(size_data / size_batch, tf.float32).eval()
            
    
        self.params = [p for layer in self.layers for p in layer.params]
        self.p_params = [p for layer in self.layers for p in layer.p_params]
        
        self.init_lr = lr
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(tf.constant(lr), dtype = tf.float32, trainable = False, name='lr')
            tf.summary.scalar('learning_rate', self.learning_rate)
        
        self.loss = self.loglike    
        if ewc:
            self.gr = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.loglike)
            for i in range(self.n_layers):
                self.loss += tf.reduce_sum(tf.square(self.gr[i][0]) * tf.square(self.params[i] - self.p_params[i]))
            
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        '''
        if ewc:
            self.train_grad = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.loss)
            self.ewc_grad = list()
            for i in range(len(self.train_grad) / 2):
                rhomin = tf.reduce_min(softplus(self.train_grad[2*i+1][1]))
                self.ewc_grad.append((self.train_grad[2*i][0] * (0.3 + softplus(self.train_grad[2*i+1][1])), self.train_grad[2*i][1])) # grad_mu * softplus(rho)
                #self.ewc_grad.append((self.train_grad[2*i][0], self.train_grad[2*i][1])) # grad_mu * softplus(rho)
                self.ewc_grad.append(self.train_grad[2*i+1])
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(self.ewc_grad)
        else:
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        '''
        
        
    def get_loss(self):
        return self.loss
    
    def get_fqpl(self, feed):
        return self.loss.eval(feed_dict = feed), self.loglike.eval(feed_dict = feed)
    
    def get_params(self):
        return self.params
    
    def print_params(self):
        for layer in self.layers:
            print "mu: {}".format(tf.reduce_mean(layer.params[0]).eval())
            print "p_mu: {}".format(tf.reduce_mean(layer.p_w).eval())
            
    def get_inputs(self):
        return [self.x, self.t]
    
    def update_prior(self):
        for layer in self.layers:
            layer.p_w.assign(layer.w).eval()
    
    '''
    def train_grads(self):
        return self.train_grad
    
    def ewc_grads(self):
        return self.ewc_grad
    
    def print_ewcgrads(self, feed):
        for i in range(len(self.ewc_grad)):
            print self.ewc_grad[i][1].name
            print "max: {}, min: {}, mean: {}, std: {}".format(np.max(self.ewc_grad[i][0].eval(feed)), np.min(self.ewc_grad[i][0].eval(feed)), np.mean(self.ewc_grad[i][0].eval(feed)), np.std(self.ewc_grad[i][0].eval(feed)))
    '''
    
    def train(self, feed):
        self.train_op.run(feed_dict = feed)
        
    def validate(self, feed):
        return self.acc.eval(feed_dict = feed)
        
    def decay_lr(self, rate_decay = 0.5, limit_decay = 1e-8):
        if self.learning_rate.eval() > limit_decay:
            self.learning_rate.assign(self.learning_rate.eval() * rate_decay).eval()
            
    def reset_lr(self):
        self.learning_rate.assign(self.init_lr).eval()
    
    def get_lr(self):
        return self.learning_rate.eval()
    