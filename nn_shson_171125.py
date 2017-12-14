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
    
    def __init__(self, model, inp, shape, sb, mu, outact = tf.sigmoid, layernum = 0, dropout = False):
        with tf.name_scope('layer' + str(layernum)):
            shape[0] += 1
            with tf.name_scope('q_pos'):
                self.w = tf.Variable(tf.truncated_normal(shape, stddev = mu), dtype = tf.float32, name = 'mu')
                variable_summaries(self.w)

            with tf.name_scope('p_pri'):
                self.p_w = tf.Variable(tf.constant(0.0, shape = shape), trainable = False, dtype = tf.float32, name = 'p_mu')
                variable_summaries(self.p_w)

            if dropout:
                inp = tf.nn.dropout(inp, model.keep_probs[layernum])
            shape_inp = tf.shape(inp)
            iones = tf.ones(shape = [shape_inp[0], 1])
            newinp = tf.concat(axis = 1, values = [inp, iones])

            self.pre_o = tf.matmul(newinp, self.w)
            self.out = outact(self.pre_o)
            tf.summary.histogram('activation', self.out)

            self.params = [self.w]
            self.p_params = [self.p_w]

            print ("layer done")

class nn_model(object):
    
    def __init__(self, shape, size_data, size_batch, mu = 0.1, outact = tf.sigmoid, seed = 1234, lr = 1e-8, ewc = False, dropout = False, l2_reg = False, reg_penalty = 5):
        
        self.n_layers = len(shape) - 1
        #self.n_samples = n_samples
        self.size_batch = size_batch
        self.num_batch = tf.placeholder(tf.float32, [1], name = 'num_batch')
        self.reg_penalty = reg_penalty
               
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name = 'x')
        self.t = tf.placeholder(tf.float32, [None, shape[-1]], name = 't')
        self.keep_probs = tf.placeholder(tf.float32, [self.n_layers], name='keep_probs')
        
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
                
            self.layers.append(nn_layer(self, inp, [shape[i], shape[i+1]], size_batch, mu, outact = actout, layernum = i, dropout = dropout))
            
        self.pred = self.layers[-1].out
        print self.pred.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        
        with tf.name_scope('terminal'):
            self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = tf.reshape(self.pred, [-1, shape[-1]]), labels = self.t), name = 'loglike')

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

        self.grads = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.loglike)
        self.gr = list()
        self.ca = list()
        self.cb = list()
        self.cc = list()
        
        for i in range(self.n_layers):
            self.gr.append(tf.Variable(tf.constant(0.0, shape = [shape[i]+1, shape[i+1]]), trainable = False, dtype = tf.float32, name = 'grads'))
            self.ca.append(tf.Variable(tf.constant(0.0, shape = [shape[i]+1, shape[i+1]]), trainable = False, dtype = tf.float32, name = 'ca'))
            self.cb.append(tf.Variable(tf.constant(0.0, shape = [shape[i]+1, shape[i+1]]), trainable = False, dtype = tf.float32, name = 'cb'))
            self.cc.append(tf.Variable(tf.constant(0.0, shape = [shape[i]+1, shape[i+1]]), trainable = False, dtype = tf.float32, name = 'cc'))
            
        self.loss = self.loglike        
        
        #self.lreg_add = tf.Variable(tf.constant(0.0, shape = []), trainable = False, dtype = tf.float32, name = 'lreg_add')
        self.loss_reg = self.loglike
        
                
        for i in range(self.n_layers):
            
            if ewc:
                self.loss_reg += (self.reg_penalty / 2.) * tf.reduce_sum(self.ca[i] * tf.square(self.params[i])
                                                                         -2 * self.params[i] * self.cb[i]
                                                                         + self.cc[i])
            #    self.lreg_add += (self.reg_penalty / 2.) * tf.reduce_sum(self.gr[i] * tf.square(self.params[i] - self.p_params[i]))
                
            
            if l2_reg: 
                self.loss_reg += (self.reg_penalty / 2.) * tf.reduce_sum(tf.square(self.params[i] - self.p_params[i]))
            
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op_reg = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_reg) 
        
        self.f_asgns = list()
        self.f_asgn_adds = list()
        self.f_avgs = list()
        #self.loss_adds = self.loss_reg.assign_add(self.lreg_add)
        self.ca_adds = list()
        self.cb_adds = list()
        self.cc_adds = list()
        self.loss_adds = list()
        
        for i in range(self.n_layers):
            self.f_asgns.append(self.gr[i].assign(tf.square(self.grads[i][0])))
            self.f_asgn_adds.append(self.gr[i].assign_add(tf.square(self.grads[i][0])))
            self.f_avgs.append(self.gr[i].assign(self.gr[i] / self.num_batch))
            
            self.ca_adds.append(self.ca[i].assign(self.gr[i]))
            self.cb_adds.append(self.cb[i].assign(self.gr[i] * self.p_params[i]))
            self.cc_adds.append(self.cc[i].assign(self.gr[i] * tf.square(self.p_params[i])))
            
            '''
            self.ca_adds.append(self.ca[i].assign_add(self.gr[i]))
            self.cb_adds.append(self.cb[i].assign_add(self.gr[i] * self.p_params[i]))
            self.cc_adds.append(self.cc[i].assign_add(self.gr[i] * tf.square(self.p_params[i])))
            '''
            
            '''
            self.loss_adds.append(self.lreg_add.assign_add((self.reg_penalty / 2.) * 
                                                           tf.reduce_sum(self.gr[i] * tf.square(self.params[i] - self.p_params[i])
                                                                        )
                                                          )
                                 )
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
    
    def update_prior(self, prior_normalize = False):
        for layer in self.layers:
            layer.p_w.assign(layer.w).eval()
    
    def calculate_fisher(self, feed, i):
        if i < 1:
            for j in range(self.n_layers):
                self.f_asgns[j].eval(feed_dict = feed)
        else:
            for j in range(self.n_layers):
                self.f_asgn_adds[j].eval(feed_dict = feed)
    
    def average_fisher(self, feed):
        for i in range(self.n_layers):
            self.f_avgs[i].eval(feed_dict = feed)
        for i in range(self.n_layers):
            self.ca_adds[i].eval()
            self.cb_adds[i].eval()
            self.cc_adds[i].eval()
            
            #self.loss_adds[i].eval()
        
    
    #def update_fisher
    
    def print_fisher(self):        
        for i in range(self.n_layers):
            print "##{} th layer's fisher##".format(i)
            print "mean: {}".format(tf.reduce_mean(self.gr[i]).eval())
            print "max: {}".format(tf.reduce_max(self.gr[i]).eval())
            print "min: {}".format(tf.reduce_min(self.gr[i]).eval())
        
    def train(self, feed, i):
        if i < 1:
            self.train_op.run(feed_dict = feed)
        else: 
            self.train_op_reg.run(feed_dict = feed)
    '''    
    def train(self, feed):
        self.train_op.run(feed_dict = feed)
    '''    
    def validate(self, feed):
        return self.acc.eval(feed_dict = feed)
        
    def decay_lr(self, rate_decay = 0.5, limit_decay = 1e-10):
        if self.learning_rate.eval() > limit_decay:
            self.learning_rate.assign(self.learning_rate.eval() * rate_decay).eval()
            
    def reset_lr(self):
        self.learning_rate.assign(self.init_lr).eval()
    
    def get_lr(self):
        return self.learning_rate.eval()
    