import tensorflow as tf
import numpy as np



def variable_summaries(var):
    with tf.name_scope('summaries'):
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

class bnn_layer(object):
    
    def __init__(self, inp, shape, sb, mu, rho, n_samples, rseed, rhoact = softplus, outact = tf.sigmoid, layernum = 0 ):
        with tf.name_scope('layer' + str(layernum)):
            shape[0] += 1
            with tf.name_scope('q_pos'):
                self.w = tf.Variable(tf.truncated_normal(shape, stddev = mu), dtype = tf.float32, name = 'mu')
                self.r = tf.Variable(tf.truncated_normal(shape, stddev = rho), dtype = tf.float32, name = 'rho')
                self.n_samples = n_samples
                variable_summaries(self.w)
                variable_summaries(self.r)

            with tf.name_scope('p_pri'):
                self.p_w = tf.Variable(tf.constant(0.0, shape = shape), trainable = False, dtype = tf.float32, name = 'p_mu')
                self.p_r = tf.Variable(tf.constant(-1.0, shape = shape), trainable = False, dtype = tf.float32, name = 'p_rho')
                self.p_bgr = tf.Variable(tf.constant(10.0, shape = shape), trainable = False, dtype = tf.float32)
                variable_summaries(self.p_w)
                variable_summaries(self.p_r)

            self.e = tf.random_normal([self.n_samples, shape[0], shape[1]], seed = rseed)
            self.w3 = tf.tile(tf.expand_dims(self.w, 0), [self.n_samples, 1, 1])
            self.r3 = tf.tile(tf.expand_dims(self.r, 0), [self.n_samples, 1, 1])
            self.p_w3 = tf.tile(tf.expand_dims(self.p_w, 0), [self.n_samples, 1, 1])                         
            self.p_r3 = tf.tile(tf.expand_dims(self.p_r, 0), [self.n_samples, 1, 1])
            self.p_bgr3 = tf.tile(tf.expand_dims(self.p_bgr, 0), [self.n_samples, 1, 1])                       

            self.c_w = self.w3 + rhoact(self.r3) * self.e
            self.q_pos = normpdf(self.c_w, self.w3, rhoact(self.r3))
            self.p_pri = 0.8 * normpdf(self.c_w, self.p_w3, rhoact(self.p_r3))\
                         + 0.2 * normpdf(self.c_w, self.p_w3, rhoact(self.p_bgr3))

            #test = tf.range(sb)
            #iones = tf.constant(1, shape = [shape[0], 1])
            iones = tf.ones(shape = tf.shape(inp))
            #print iones.get_shape()
            #print inp.get_shape()
            iones2 = tf.reduce_mean(iones, 2, keep_dims = True)
            newinp = tf.concat(2, [inp, iones2])
            #print newinp.get_shape()


            self.pre_o = tf.batch_matmul(newinp, self.c_w)
            self.out = outact(self.pre_o)
            tf.summary.histogram('activation', self.out)

            #self.log_q_pos = tf.reduce_sum(tf.log(tf.clip_by_value(self.q_pos, 1e-20, 1e+20)), [1, 2])
            #self.log_p_pri = tf.reduce_sum(tf.log(tf.clip_by_value(self.p_pri, 1e-20, 1e+20)), [1, 2])
            self.log_q_pos = tf.reduce_sum(tf.log(tf.clip_by_value(self.q_pos, 1e-20, 1e+20)))
            self.log_p_pri = tf.reduce_sum(tf.log(tf.clip_by_value(self.p_pri, 1e-20, 1e+20)))
            tf.summary.scalar('log_q_pos', self.log_q_pos)
            tf.summary.scalar('log_p_pri', self.log_p_pri)

            # self.log_q_pos = tf.reduce_sum(tf.log(self.q_pos), [1, 2])
            # self.log_p_pri = tf.reduce_sum(tf.log(self.p_pri), [1, 2])

            self.params = [self.w, self.r]
            self.p_params = [self.p_w, self.p_r]

            print ("layer done")

class bnn_model(object):
    
    def __init__(self, shape, size_data, size_batch, mu = 0.1, rho = 0.1, n_samples = 10, outact = tf.sigmoid, seed = 1234, lr = 1e-8, kl_reweight = True):
        
        self.n_layers = len(shape) - 1
        self.n_samples = n_samples
        
        #self.x = tf.placeholder(tf.float32, [size_batch, shape[0]])
        #self.t = tf.placeholder(tf.float32, [size_batch, shape[-1]])
       
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name = 'x')
        self.t = tf.placeholder(tf.float32, [None, shape[-1]], name = 't')
            
        #size_batch = tf.to_int32(tf.shape(self.x)[0])
        
        self.x3 = tf.tile(tf.expand_dims(self.x, 0), [self.n_samples, 1, 1])
        
        self.layers = list()
        
        for i in range(self.n_layers):
            if i == 0:
                inp = self.x3
            else:
                inp = self.layers[i-1].out
                
            if i == self.n_layers-1:
                #actout = tf.nn.softmax
                #actout = fsoftmax
                actout = dummyfunc
            else: 
                actout = outact
                
            self.layers.append(bnn_layer(inp, [shape[i], shape[i+1]], size_batch, mu, rho, n_samples = self.n_samples, rseed = seed + i, outact = actout, layernum = i))
            
        self.pred = self.layers[-1].out
        print self.pred.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        
        '''
        tr1 = tf.range(tf.to_int32(tshape[0]))
        tr2 = tr1 * tf.to_int32(tshape[1]) + tf.to_int32(tmaxs)
        tr3 = tf.tile(tr2, [self.n_samples])
        tr4 = tf.gather(tf.reshape(self.pred, [-1]), tr3)
        tr5 = tf.reshape(tr4, [self.n_samples, -1])
        '''
      
        #self.loglike = tf.reduce_sum(tf.log(tf.clip_by_value(tr5, 1e-20, 1e+20)), [1])
        
        with tf.name_scope('terminal'):
            self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(self.pred, [-1, shape[-1]]), tf.tile(self.t, [self.n_samples, 1])), name = 'loglike')

            self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.reduce_mean(self.pred, [0]), 1), tf.argmax(self.t, 1))), name = 'accuracy')

            self.log_q_pos = tf.reduce_sum([layer.log_q_pos for layer in self.layers], name = 'total_lqpos')
            self.log_p_pri = tf.reduce_sum([layer.log_p_pri for layer in self.layers], name = 'total_lppri')
            
            tf.summary.scalar('loglike', self.loglike)
            tf.summary.scalar('log_q_pos', self.log_q_pos)
            tf.summary.scalar('log_p_pri', self.log_p_pri)
            tf.summary.scalar('accuracy', self.acc)
            
            #print self.loglike.get_shape()
            #print self.log_q_pos.get_shape()
     
        self.n_batches = tf.cast(size_data / size_batch, tf.float32).eval()
        self.init_kl = tf.cast(2.0**self.n_batches / (2.0**self.n_batches - 1.0), tf.float32).eval()
        
        #self.coeff_kl = tf.cast(size_batch / size_data, tf.float32)
        self.coeff_kl = tf.Variable(tf.constant(self.init_kl), trainable = False, dtype = tf.float32)
        
        if kl_reweight:
            #self.loss = tf.reduce_mean((self.log_q_pos - self.log_p_pri) * self.coeff_kl - self.loglike, [0])
            self.loss = (self.log_q_pos - self.log_p_pri) * self.coeff_kl + self.loglike
        else:            
            #self.loss = tf.reduce_mean(self.log_q_pos - self.log_p_pri - self.loglike, [0])
            self.loss = self.log_q_pos - self.log_p_pri + self.loglike
    
        self.params = [p for layer in self.layers for p in layer.params]
        self.p_params = [p for layer in self.layers for p in layer.p_params]
        
        self.init_lr = lr
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(tf.constant(lr), dtype = tf.float32)
            tf.summary.scalar('learning_rate', self.learning_rate)
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_loss(self):
        return self.loss
    
    def get_fqpl(self, feed):
        return self.loss.eval(feed_dict = feed), self.log_q_pos.eval(feed_dict = feed), self.log_p_pri.eval(feed_dict = feed), self.loglike.eval(feed_dict = feed)
    
    def get_params(self):
        return self.params
    
    def get_inputs(self):
        return [self.x, self.t]
    
    def update_prior(self):
        for layer in self.layers:
            layer.p_w.assign(layer.w.eval())
            layer.p_r.assign(layer.r.eval())
    
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
    
    def decay_klrw(self):
        self.coeff_kl.assign(self.coeff_kl.eval() * 0.5).eval()
    
    def reset_klrw(self):
        self.coeff_kl.assign(self.init_kl).eval()
        
    def get_klrw(self):
        return self.coeff_kl.eval()
        
    