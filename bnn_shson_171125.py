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

class bnn_layer(object):
    
    def __init__(self, inp, shape, sb, mu, rhos, n_samples, rseed, rhoact = softplus, outact = tf.sigmoid, layernum = 0, train_rho = True):
        with tf.name_scope('layer' + str(layernum)):
            shape[0] += 1
            self.shape = shape
            with tf.name_scope('q_pos'):
                self.w = tf.Variable(tf.truncated_normal(shape, stddev = mu), dtype = tf.float32, name = 'mu')
                #self.r = tf.Variable(tf.truncated_normal(shape, stddev = rhos[0]), dtype = tf.float32, name = 'rho', trainable = train_rho)
                self.r = tf.Variable(tf.constant(rhos[0], shape = shape), dtype = tf.float32, name = 'rho', trainable = train_rho)
                self.n_samples = n_samples
                variable_summaries(self.w)
                variable_summaries(self.r)

            with tf.name_scope('p_pri'):
                self.p_w = tf.Variable(tf.constant(0.0, shape = shape), trainable = False, dtype = tf.float32, name = 'p_mu')
                self.p_r = tf.Variable(tf.constant(rhos[1], shape = shape), trainable = False, dtype = tf.float32, name = 'p_rho')
                self.p_bgr = tf.Variable(tf.constant(rhos[2], shape = shape), trainable = False, dtype = tf.float32)
                variable_summaries(self.p_w)
                variable_summaries(self.p_r)

            self.e = tf.random_normal([self.n_samples, shape[0], shape[1]], seed = rseed)           
            self.c_w = self.w + rhoact(self.r) * self.e
            self.q_pos = normpdf(self.c_w, self.w, rhoact(self.r))
            self.p_pri = 0.8 * normpdf(self.c_w, self.p_w, rhoact(self.p_r))\
                         + 0.2 * normpdf(self.c_w, self.p_w, rhoact(self.p_bgr))

            shape_inp = tf.shape(inp)
            iones = tf.ones(shape = [shape_inp[0], shape_inp[1], 1])
            newinp = tf.concat(axis = 2, values = [inp, iones])

            self.pre_o = tf.matmul(newinp, self.c_w)
            self.out = outact(self.pre_o)
            tf.summary.histogram('activation', self.out)

            self.log_q_pos = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.q_pos, 1e-20, 1e+20)), [1, 2]))
            self.log_p_pri = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(self.p_pri, 1e-20, 1e+20)), [1, 2]))
            tf.summary.scalar('log_q_pos', self.log_q_pos)
            tf.summary.scalar('log_p_pri', self.log_p_pri)

            self.params = [self.w, self.r]
            self.p_params = [self.p_w, self.p_r]

            print ("layer done")

class bnn_model(object):
    
    def __init__(self, shape, size_data, size_batch, mu = 0.1, rhos = [0.1, 1.0, 10.0], n_samples = 10, outact = tf.sigmoid, seed = 1234, lr = 1e-8, kl_reweight = True, train_rho = True, only_loglike = False, ewc = False, squared_std = False, pri_type = 0, pri_coeff = 1):
        
        self.n_layers = len(shape) - 1
        self.n_samples = n_samples
        self.pri_type = pri_type
               
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name = 'x')
        self.t = tf.placeholder(tf.float32, [None, shape[-1]], name = 't')
        
        self.rhos = rhos
        self.mu = mu
        self.shape = shape
        
        self.x3 = tf.tile(tf.expand_dims(self.x, 0), [self.n_samples, 1, 1])
        
        self.layers = list()
        
        for i in range(self.n_layers):
            if i == 0:
                inp = self.x3
            else:
                inp = self.layers[i-1].out
                
            if i == self.n_layers-1:
                actout = dummyfunc
            else: 
                actout = outact
                
            self.layers.append(bnn_layer(inp, [shape[i], shape[i+1]], size_batch, mu, rhos, n_samples = self.n_samples, rseed = seed + i, outact = actout, layernum = i, train_rho = train_rho))
            
        self.pred = self.layers[-1].out
        print self.pred.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        
        with tf.name_scope('terminal'):
            self.loglike = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = tf.reshape(self.pred, [-1, shape[-1]]), labels = tf.tile(self.t, [self.n_samples, 1])), name = 'loglike')

            self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.reduce_mean(self.pred, [0]), 1), tf.argmax(self.t, 1))), name = 'accuracy')

            self.log_q_pos = tf.reduce_sum([layer.log_q_pos for layer in self.layers], name = 'total_lqpos')
            self.log_p_pri = tf.reduce_sum([layer.log_p_pri for layer in self.layers], name = 'total_lppri')
            
            tf.summary.scalar('loglike', self.loglike)
            tf.summary.scalar('log_q_pos', self.log_q_pos)
            tf.summary.scalar('log_p_pri', self.log_p_pri)
            tf.summary.scalar('accuracy', self.acc)
            
        self.n_batches = tf.cast(size_data / size_batch, tf.float32).eval()
        self.init_kl = tf.cast(2.0**self.n_batches / (2.0**self.n_batches - 1.0), tf.float32).eval()
        self.coeff_kl = tf.Variable(tf.constant(self.init_kl), trainable = False, dtype = tf.float32)
        
        if only_loglike:
            self.loss = self.loglike
        elif kl_reweight:
            #self.loss = tf.reduce_mean((self.log_q_pos - self.log_p_pri) * self.coeff_kl - self.loglike, [0])
            self.loss = (self.log_q_pos - self.log_p_pri) * self.coeff_kl + self.loglike
        else:            
            #self.loss = tf.reduce_mean(self.log_q_pos - self.log_p_pri - self.loglike, [0])
            self.loss = self.log_q_pos - self.log_p_pri + self.loglike
    
        self.params = [p for layer in self.layers for p in layer.params]
        self.p_params = [p for layer in self.layers for p in layer.p_params]
        
        self.update_pw = list()
        self.update_pr = list()
        self.update_pr_alt = list()
        
        for layer in self.layers:
            self.update_pw.append(layer.p_w.assign(layer.w))
            self.update_pr.append(layer.p_r.assign(layer.r))
            if self.pri_type == 1:
                self.update_pr_alt.append(layer.p_r.assign(pri_coeff * (layer.r - (tf.reduce_max(layer.r) + tf.reduce_min(layer.r))/2) / (tf.reduce_max(layer.r) - tf.reduce_min(layer.r))))
            elif self.pri_type == 2:
                self.update_pr_alt.append(layer.p_r.assign(layer.r + pri_coeff))
        
        self.init_lr = lr
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(tf.constant(lr), dtype = tf.float32, trainable = False, name='lr')
            tf.summary.scalar('learning_rate', self.learning_rate)
            
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
        if ewc:
            self.train_grad = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(self.loss)
            print self.train_grad
            self.ewc_grad = list()
            for i in range(len(self.train_grad) / 2):
                
                if squared_std:
                    self.ewc_grad.append((self.train_grad[2*i][0] * (softplus(self.p_params[2*i+1]))**2, self.train_grad[2*i][1])) # grad_mu * softplus(prior_rho)**2
                else: 
                    self.ewc_grad.append((self.train_grad[2*i][0] * softplus(self.p_params[2*i+1]), self.train_grad[2*i][1])) # grad_mu * softplus(prior_rho)
                
                self.ewc_grad.append(self.train_grad[2*i+1])
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).apply_gradients(self.ewc_grad)
        else:
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_loss(self):
        return self.loss
    
    def get_fqpl(self, feed):
        return self.loss.eval(feed_dict = feed), self.log_q_pos.eval(feed_dict = feed), self.log_p_pri.eval(feed_dict = feed), self.loglike.eval(feed_dict = feed)
    
    def get_params(self):
        return self.params
    
    def print_params(self, f):
        for layer in self.layers:
            m = layer.w.eval()
            r = layer.r.eval()
            pm = layer.p_w.eval()
            pr = layer.p_r.eval()
            layer_str = "\n###layer###\n"
            m_str = "mu: max {}, min {}, mean {}, std {}\n".format(np.max(m), np.min(m), np.mean(m), np.std(m))
            r_str = "rho: max {}, min {}, mean {}, std {}\n".format(np.max(r), np.min(r), np.mean(r), np.std(r))
            pm_str = "p_mu: max {}, min {}, mean {}, std {}\n".format(np.max(pm), np.min(pm), np.mean(pm), np.std(pm))
            pr_str = "p_rho: max {}, min {}, mean {}, std {}\n".format(np.max(pr), np.min(pr), np.mean(pr), np.std(pr))
            print layer_str
            print m_str
            print r_str
            #print "p_mu: {}".format(tf.reduce_mean(layer.p_params[0]).eval())
            #print "p_rho: {}".format(tf.reduce_mean(layer.p_params[1]).eval()
            print pm_str
            print pr_str
            f.write(layer_str)
            f.write(m_str)
            f.write(r_str)
            f.write(pm_str)
            f.write(pr_str)
            
    def get_inputs(self):
        return [self.x, self.t]
    
    def update_prior(self, pri_type = 0):
        for i in range(self.n_layers):
            #layer.p_w.assign(layer.w).eval()
            self.update_pw[i].eval()
            if self.pri_type == 0:
                self.update_pr[i].eval()
                # (r - min(r)) / (max(r) - min(r))
            else:
                self.update_pr_alt[i].eval()
    
    def reset_q_params(self):
        for layer in self.layers:
            layer.r.assign(tf.constant(self.rhos[0], shape = layer.shape)).eval()
    
    def train_grads(self):
        return self.train_grad
    
    def ewc_grads(self):
        return self.ewc_grad
    
    def print_ewcgrads(self, feed):
        for i in range(len(self.ewc_grad)):
            print self.ewc_grad[i][1].name
            print "max: {}, min: {}, mean: {}, std: {}".format(np.max(self.ewc_grad[i][0].eval(feed)), np.min(self.ewc_grad[i][0].eval(feed)), np.mean(self.ewc_grad[i][0].eval(feed)), np.std(self.ewc_grad[i][0].eval(feed)))
    
    def train(self, feed):
        self.train_op.run(feed_dict = feed)
        
    def validate(self, feed):
        return self.acc.eval(feed_dict = feed)
        
    def decay_lr(self, rate_decay = 0.5, limit_decay = 1e-10):
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
        
    