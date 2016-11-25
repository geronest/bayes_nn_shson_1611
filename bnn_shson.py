import tensorflow as tf
import numpy as np


def softplus(x):
    return tf.log(1 + tf.exp(x))

def normpdf(x, m, r):
    return (1 / tf.sqrt(2 * np.pi * (r**2))) * tf.exp(-(x - m)**2/2/r**2)

def fsoftmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), 2, keep_dims = True)

class bnn_layer(object):
    
    def __init__(self, inp, shape, mu, rho, n_samples, rseed, rhoact = softplus, outact = tf.sigmoid ):
        shape[0] += 1
        self.w = tf.Variable(tf.truncated_normal(shape, stddev = mu), dtype = tf.float32)
        self.r = tf.Variable(tf.truncated_normal(shape, stddev = rho), dtype = tf.float32)
        self.n_samples = n_samples
        
        self.p_w = tf.Variable(tf.constant(0.0, shape = shape), trainable = False, dtype = tf.float32)
        self.p_r = tf.Variable(tf.constant(-1.0, shape = shape), trainable = False, dtype = tf.float32)
        self.p_bgr = tf.Variable(tf.constant(1.0, shape = shape), trainable = False, dtype = tf.float32)
        
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
        #self.pre_o = tf.matmul(inp, self.c_w[:, :-1, :]) + tf.transpose(self.c_w[:,-1,:], [1, 0, 2])
        
        #test1 = tf.batch_matmul(inp, self.c_w[:, :-1, :])
        #test2 = tf.transpose(self.c_w[:,-1,:], [1, 0, 2])
        
        #print tf.shape(test2)
        #self.pre_o = tf.batch_matmul(inp, self.c_w[:, :-1, :]) + tf.transpose(self.c_w[:,-1,:], [1, 0, 2])

        
        
        iones = tf.constant(1, shape = [shape[0], 1])
        newinp = tf.concat(1, [inp, iones])
        
        self.pre_o = tf.batch_matmul(newinp, self.c_w)
        
        self.out = outact(self.pre_o)
                                     
        self.log_q_pos = tf.reduce_sum(tf.log(self.q_pos), [1, 2])
        self.log_p_pri = tf.reduce_sum(tf.log(self.p_pri), [1, 2])
        
        self.params = [self.w, self.r]
        self.p_params = [self.p_w, self.p_r]
        
        print ("layer done")

class bnn_model(object):
    
    def __init__(self, shape, size_batch, mu = 0.1, rho = 0.1, n_samples = 10, outact = tf.sigmoid, seed = 1234, lr = 1e-8):
        
        self.n_layers = len(shape) - 1
        self.n_samples = n_samples
        
        #self.x = tf.placeholder(tf.float32, [size_batch, shape[0]])
        #self.t = tf.placeholder(tf.float32, [size_batch, shape[-1]])
        
        self.x = tf.placeholder(tf.float32, [None, shape[0]])
        self.t = tf.placeholder(tf.float32, [None, shape[-1]])
        
        self.x3 = tf.tile(tf.expand_dims(self.x, 0), [self.n_samples, 1, 1])
        
        self.layers = list()
        
        for i in range(self.n_layers):
            if i == 0:
                inp = self.x3
            else:
                inp = self.layers[i-1].out
                
            if i == self.n_layers-1:
                #actout = tf.nn.softmax(dim = 2)
                actout = fsoftmax
            else: 
                actout = outact
                
            self.layers.append(bnn_layer(inp, [shape[i], shape[i+1]], mu, rho, n_samples = self.n_samples, rseed = seed + i, outact = actout))
            
        self.pred = self.layers[-1].out
        print self.t.get_shape()
        
        tmaxs = tf.argmax(self.t, 1)
        tshape = tf.shape(self.t)
        
        tr1 = tf.range(tf.to_int32(tshape[0]))
        tr2 = tr1 * tf.to_int32(tshape[1]) + tf.to_int32(tmaxs)
        tr3 = tf.tile(tr2, [self.n_samples])
        tr4 = tf.gather(tf.reshape(self.pred, [-1]), tr3)
        tr5 = tf.reshape(tr4, [self.n_samples, -1])
      
        self.loglike = tf.reduce_sum(tf.log(tr5), [1])
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf.reduce_mean(self.pred, [0]), 1), tf.argmax(self.t, 1))))
       
        self.log_q_pos = tf.reduce_sum([layer.log_q_pos for layer in self.layers])
        self.log_p_pri = tf.reduce_sum([layer.log_p_pri for layer in self.layers])
        
        self.loss = tf.reduce_mean(self.log_q_pos - self.log_p_pri - self.loglike, [0])
        self.params = [p for layer in self.layers for p in layer.params]
        self.p_params = [p for layer in self.layers for p in layer.p_params]
        
        self.learning_rate = lr
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_loss(self):
        return self.loss
    
    def get_params(self):
        return self.params
    
    def get_inputs(self):
        return [self.x, self.t]
    
    def update_prior(self):
        for layer in self.layers:
            layer.p_w = layer.w
            layer.p_r = layer.r
    
    def train(self, feed):
        self.train_op.run(feed_dict = feed)
        
    def validate(self, feed):
        return self.acc.eval(feed_dict = feed)
    
    
        
        
'''
class NNModel(object):
    # dims: list of nnlayer dimensions (example: [784, 50, 10] for MNIST with one hidden layer
    # n_samples: number of samples used for one input instance. the larger, the slower.
    # activation: activation function used for output of each nnlayer
    # seed: random seed
	def __init__(self, dims, n_samples=100, activation=T.nnet.sigmoid, seed=1234):
		assert dims[-1] != 2 # no 2-dim softmax
		self.dims = dims
        # is_binary: indicates whether the output of the entire network is binary or not
		is_binary = (dims[-1] == 1)
        # n_layers: number of layers used for this network
		self.n_layers = len(dims)-1
		assert self.n_layers >= 1

        # x: input vector
        # t: target vector
        # x3d: input vector copied n_sample times
		self.x = T.matrix('x')
		self.t = T.ivector('t')
		x3d = self.x.dimshuffle('x',0,1).repeat(n_samples, axis=0)

        # nns: list of nnlayers
		self.nns = list()
        
        # building layers
		for l in xrange(self.n_layers):
			if l==0:
				inp = x3d # the bottom layer of the network. x3d is used for the input of this layer.
			else:
				inp = self.nns[l-1].output # otherwise, output from the last layer will be used as the input for this layer.

			if l == self.n_layers-1: # the top layer of the network. 
				if is_binary:
					act = activation
				else:
					act = lambda x: log_softmax(x, axis=2) # if the output is not binary, log_softmax is used.
			else:
				act = activation # if it is not the top layer, then 'act' will be used for its activation function.
				
            # adds new nnlayer to nns
			self.nns.append( NNLayer(dims[l], dims[l+1], inp, n_samples=n_samples, seed=seed+l, activation=act, nick=str(l)) )

        # prob_y: output vector 
		self.prob_y = self.nns[-1].output
		if is_binary:
            # t3d: target vector copied n_sample times
			t3d = self.t.dimshuffle('x',0,'x')
            # loglike: log likelihood
			self.loglike = T.sum( t3d*T.log(self.prob_y) + (1-t3d)*T.log(1-self.prob_y) , axis=[1,2])
            # acc: accuracy calculated by using target vector
			self.acc = T.mean( T.eq( (T.mean(self.prob_y, axis=0)>0.5)[:,0], self.t) )
		else:
			self.loglike = T.sum( self.prob_y[:,T.arange(self.t.shape[0]),self.t], axis=1)
			self.acc = T.mean( T.eq( T.argmax(T.mean(self.prob_y, axis=0), axis=1), self.t) )
		

        # log_q_posterior: log(probability of variational posterior(weight))
        # log_p_prior: log(probability of prior(weight))
		self.log_q_posterior = sum([nn.log_q_posterior for nn in self.nns])
		self.log_p_prior = sum([nn.log_p_prior for nn in self.nns])

        # f: loss function value (log(q(w)) - log(p(w)) - log(p(D|w)) )
        # params: parameters (mu and rho) used for variational posterior distribution
        # prior_params: parameters (mu and spike_rho) used for prior distribution
		self.f = T.mean(self.log_q_posterior - self.log_p_prior - self.loglike, axis=0)
		self.params = [param for nn in self.nns for param in nn.params]
		self.prior_params = [pp for nn in self.nns for pp in nn.prior_params]
	
    # initialize(reset)s the parameters of the given network's variational posterior.
    # Only used for logistic regression of toy data
	def init_params(self, mu_scale=0.01, rho_scale=0.05):
		for nn in self.nns:
			nn.init_params(mu_scale, rho_scale)

	def get_loss(self):
		return self.f

	def get_params(self):
		return self.params

	def get_inputs(self):
		return [self.x, self.t]
'''



        
                                     
                                 
                                 