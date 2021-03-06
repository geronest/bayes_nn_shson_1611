import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams.update({'figure.autolayout': True, 'figure.subplot.bottom' : 0.15})
matplotlib.rcParams.update({'figure.subplot.bottom' : 0.15})

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from bnn_shson_171125 import *
import nn_shson_171125 as nn_shson
from shson_exp_manager import *
import h5py
import random


def num_to_onehot(nums, n_labels):
    results = list()
    for i in range(len(nums)):
        res = np.zeros([n_labels])
        res[nums[i]] = 1
        results.append(res)
    return np.asarray(results, dtype = 'float32')

def print_accs(accs, ep):
    res = ""
    for acc in accs:
        res += " {:.4f}".format(acc[ep])
    
    print res
    
def last_accs(accs, i):
    res = list()
    for j in range(i):
        try:
            res.append(accs[j][-1])
        except:
            res.append(0.)
    '''    
    for acc in accs:
        res.append(acc[-1])
    '''
    return res


class exp_manager(object):
    
    def __init__(self, data, num_tasks = 7, num_labels = 10, index_data = [], rseed = 1337, batch_size = 100, name_dir = "supgrad"):
        random.seed(rseed)
        self.data = data
        self.index_data = index_data
        self.num_tasks = num_tasks
        self.x_train = list()
        self.x_valid = list()
        self.x_test = list()
        self.t_train = num_to_onehot(self.data['train_label'][()], num_labels) 
        self.t_valid = num_to_onehot(self.data['valid_label'][()], num_labels) 
        self.t_test = num_to_onehot(self.data['test_label'][()], num_labels) 
        self.batch_size = batch_size
        self.model_specs = list()
        self.valids_per_task = list()
        self.avg_perf = list()
        self.avg_perf_task = list()
        self.task_ends = list()
        
        self.model = None
        self.sess = None
        self.merged = None
        self.train_writer = None
        self.test_writer = None
        
        self.name_dir = name_dir
        self.savedir = make_savedir("experiment_saves_{}/".format(name_dir))

        
        
    def multiply_data(self, data, num_tasks, resolution, index_data = []):
        perm = range(resolution)
        
        for i in range(num_tasks):
            self.x_train.append(data['train_data'][()][:, perm])
            self.x_valid.append(data['valid_data'][()][:, perm])
            self.x_test.append(data['test_data'][()][:, perm])
            random.shuffle(perm)
            
            if i < 2:
                print perm[0:20]
            
        print ("multiply_data with {} tasks done.".format(num_tasks))
        
    
    def dist_data(self, data_x, data_t, sp_labels, n_labels = 10):
        num_split = len(sp_labels)
        res_x = list()
        res_t = list()

        for i in range(num_split):
            res_x.append(list())
            res_t.append(list())

        for i in range(len(data_t)):
            ti = data_t[i]
            for j in range(num_split):
                if ti in sp_labels[j]: # label found in sp_labels[j], add this element
                    res_x[j].append(data_x[i])
                    res_t[j].append(ti)

        res_x = np.array(res_x)
        res_t = np.array(res_t)

        for i in range(num_split):
            res_t[i] = num_to_onehot(res_t[i], n_labels)

        return res_x, res_t

    def split_data(self, data, label_per_split = 2, num_labels = 10, resolution = 784, index_data = []):

        num_split = num_labels / label_per_split
        split_labels = list()
        slabel = 0
        for i in range(num_split):
            split_labels.append(list())
            for j in range(label_per_split):
                split_labels[-1].append(slabel)
                slabel += 1
        print split_labels

        self.x_train, self.t_train = self.dist_data(data['train_data'][()], data['train_label'][()], split_labels, num_labels)
        self.x_valid, self.t_valid = self.dist_data(data['valid_data'][()], data['valid_label'][()], split_labels, num_labels)
        self.x_test, self.t_test = self.dist_data(data['test_data'][()], data['test_label'][()], split_labels, num_labels)    
        
    def init_session(self):
        try:
            tf.reset_default_graph()
            self.sess.close()
        except:
            pass
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, \
                                                           gpu_options=tf.GPUOptions(allow_growth = True)))
        
    def assign_model(self, model):
        self.model = model
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.savedir + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.savedir + 'test')
        
    # kind_model is either 'bnn' or 'nn'
    def train(self, name_train, n_epochs, patience = 3, vdec_range = 20, fs_decay = False, fs_range = 25,\
              update_prior = False, dropout = False, ewc = False, l2_reg = False, supgrad = False, reset_q_params = False, pri_type = 0, kind_model = 'bnn', kind_prob = 'permute'): # kind_prob is either 'permute' or 'split'
        
        name_rec = self.savedir + "train_" + name_train + get_timedir()
        file_rec = open(name_rec + ".txt", 'w')
        time_train_start = record_time()
        
        msg_taskskip = "=== cannot decay more. stop learning this task ===\n"
        
        self.model_specs.append(list()) ## TODO
        self.valids_per_task.append(list())
        self.avg_perf.append(list())
        self.avg_perf_task.append(list())
        self.task_ends.append(list())

        tf.global_variables_initializer().run()

        n_datas = self.num_tasks

        n_epochs = n_epochs
        n_batches = list()
        for i in range(self.num_tasks):
            if kind_prob == 'permute': n_batches.append(len(self.t_train) / self.batch_size)
            elif kind_prob == 'split': n_batches.append(len(self.t_train[i]) / self.batch_size)
        patience_exp = patience
        acc_ends = 0
        
        fs = list()
        qs = list()
        ps = list()
        ls = list()
        fs_mean = list()
        taccs = list()
        taccs_mean = list()
        vaccs = list()
        test_accs = list()
        break_ep = False
        for i in range(n_datas):
            vaccs.append(list())

        for d in range(n_datas):
            self.model.reset_lr()
            
            #fs_mean = list()
            
            for ep in range(n_epochs):
                self.x_train, self.t_train = shuffle_data(self.x_train, self.t_train, kind_prob) # SGD
                '''
                print "###### Data examining ######"
                for i in range(len(self.x_train)):
                    print "length of {}th data: {} {}".format(i, len(self.x_train[i]), len(self.t_train[i]))
                print "###### Data examining done ######"                                                            
                '''
                
                for i in range(n_batches[d]):
                    break_ep = False
                    
                    if kind_prob == 'permute': 
                        feed = {self.model.x: self.x_train[d][i*batch_size:(i+1)*batch_size], \
                                self.model.t: self.t_train[i*batch_size:(i+1)*batch_size]}
                    elif kind_prob == 'split': 
                        feed = {self.model.x: self.x_train[d][i*batch_size:(i+1)*batch_size], \
                                self.model.t: self.t_train[d][i*batch_size:(i+1)*batch_size]}
                    #print "### len_xtrain: {}, len_ttrain: {}".format(len(self.x_train[d][i*batch_size:(i+1)*batch_size]), len(self.t_train[d][i*batch_size:(i+1)*batch_size]))
                        
                    #if kind_model == 'bnn':
                        #v_f, v_q, v_p, v_l = self.model.get_fqpl(feed)
                        #fs.append(v_f), qs.append(v_q), ps.append(v_p), ls.append(v_l)

                    if kind_model == 'bnn':
                        self.model.train(feed)
                    elif kind_model == 'nn':
                        if ewc:
                            self.model.train(feed, d)
                        elif l2_reg: 
                            self.model.train(feed, 1)
                        else:
                            self.model.train(feed, 0)
                    
                    if (i % 50 == 0) and (ep % 50 == 0):
                        train_accuracy = self.model.validate(feed)

                        print("ep %d, batch %d, training accuracy %g"%(ep, i, train_accuracy))
                        #if kind_model == 'bnn': print("f : {}, q : {}, p : {}, l : {}".format(v_f, v_q, v_p, v_l))
                    '''    
                    if fs_decay and i > fs_range and np.mean(fs[-fs_range:]) < fs[-1]:
                        if patience == 0:
                            last_lr = self.model.get_lr()
                            self.model.decay_lr()
                            patience = 3

                            if self.model.get_lr() == last_lr:
                                print(msg_taskskip)
                                file_rec.write(msg_taskskip)
                                break_ep = True
                                self.valids_per_task[-1].append(last_accs(vaccs, n_datas))
                                self.avg_perf_task[-1].append(np.mean(np.array(last_accs(vaccs, d+1))))
                                break
                        else:
                            patience -= 1
                    '''

                # fs_mean.append(np.mean(fs[-n_batches:]))
                
                if supgrad and (ep % 50 == 0): self.model.print_ewcgrads(feed)

                str_vacc = "data %d, ep %d, valid accuracy:"%(d, ep)
                for i in range(n_datas): 
                    if kind_prob == 'permute': 
                        vaccs[i].append(self.model.validate({self.model.x: self.x_valid[i], self.model.t: self.t_valid}))
                    elif kind_prob == 'split': 
                        vaccs[i].append(self.model.validate({self.model.x: self.x_valid[i], self.model.t: self.t_valid[i]}))
                    str_vacc += " {:.5g}".format(vaccs[i][-1])
                    
                avg_vacc = np.mean(np.array(last_accs(vaccs, d+1)))    
                str_vacc += ", avg: {:.5g}".format(avg_vacc)

                self.avg_perf[-1].append(avg_vacc)
                taccs.append(train_accuracy)
                print(str_vacc)
                file_rec.write(str_vacc + "\n")
                
                if kind_prob == 'permute': 
                    summary = self.sess.run(self.merged, feed_dict ={self.model.x: self.x_valid[d], self.model.t: self.t_valid})
                elif kind_prob == 'split': 
                    summary = self.sess.run(self.merged, feed_dict ={self.model.x: self.x_valid[d], self.model.t: self.t_valid[d]})
                    
                self.test_writer.add_summary(summary, (d+1)*(ep+1))
                
                if break_ep: break
                if ep > (vdec_range * 2) and np.mean(vaccs[d][-(vdec_range * 2):-vdec_range]) >= np.mean(vaccs[d][-vdec_range:]):
                    if patience_exp == 0: 
                        last_lr = self.model.get_lr()
                        self.model.decay_lr()
                        patience_exp = patience
                        
                        if self.model.get_lr() == last_lr:
                            print(msg_taskskip)
                            file_rec.write(msg_taskskip)
                            self.valids_per_task[-1].append(last_accs(vaccs, n_datas))
                            self.avg_perf_task[-1].append(np.mean(np.array(last_accs(vaccs, d+1))))
                            acc_ends += (ep+1)
                            self.task_ends[-1].append(acc_ends)
                            break_ep = True
                            break
                    else:
                        patience_exp -= 1      
                        

            if not break_ep: 
                self.valids_per_task[-1].append(last_accs(vaccs, n_datas))
                self.avg_perf_task[-1].append(np.mean(np.array(last_accs(vaccs, d+1))))
                acc_ends += (ep+1)
                self.task_ends[-1].append(acc_ends)
            
            if kind_model == 'nn' and ewc:
                for i in range(n_batches[d]):
                    if kind_prob == 'permute': 
                        feed = {self.model.x: self.x_train[d][i*batch_size:(i+1)*batch_size], \
                                self.model.t: self.t_train[i*batch_size:(i+1)*batch_size]}
                    elif kind_prob == 'split': 
                        feed = {self.model.x: self.x_train[d][i*batch_size:(i+1)*batch_size], \
                                self.model.t: self.t_train[d][i*batch_size:(i+1)*batch_size]}
                    self.model.calculate_fisher(feed, i)
                
                self.model.average_fisher({self.model.num_batch: [n_batches[d]]})
                self.model.print_fisher()
                
            if update_prior: self.model.update_prior(pri_type)
            if reset_q_params: self.model.reset_q_params()
        
        time_train = elapsed_time(time_train_start)
        file_rec.write("Training time: {} seconds\n".format(time_train))
        np.save(name_rec + "_avgperf", np.array(self.avg_perf_task[-1]))
        np.save(name_rec + "_taskperf", np.array(self.valids_per_task[-1]))
        
        # print accuracies where each task ended
        task_valids = "\n\n {} \n\n".format(self.valids_per_task[-1])
        print task_valids
        file_rec.write(task_valids)
        
        
        # Test
        time_test_start = record_time()
        
        str_testacc = "\n#####\ntest accuracy: "
        for i in range(n_datas):
            if kind_prob == 'permute': 
                test_accs.append(self.model.validate({self.model.x: self.x_test[i], self.model.t: self.t_test}))
            elif kind_prob == 'split':
                test_accs.append(self.model.validate({self.model.x: self.x_test[i], self.model.t: self.t_test[i]}))
            str_testacc += " {:.5g}".format(test_accs[-1])
        avg_testacc = np.mean(np.array(test_accs))    
        str_testacc += ", avg: {:.5g}".format(avg_testacc)
        print (str_testacc)
        file_rec.write(str_testacc + "\n")
        
        time_test = elapsed_time(time_test_start)
        file_rec.write("Test time: {} seconds\n".format(time_test))
        file_rec.write("#####\n")
        
        # Plot
        colors = ['b', 'g', 'r', 'c']
        markers = ['o', '^', 'd', '*', 's']
        lss = ['-', '--', '-.']
        legends = list()
        fig = plt.figure(figsize=(10, 10))
        #plt.title("Valid performances, " + name_train)      
        plt.title("Validation performance of each task")      
        valpertask = np.array(self.valids_per_task[-1])
        for i in range(n_datas):
            x_plot = range(1, self.num_tasks+1)
            plt.plot(x_plot, valpertask[:, i], colors[(i % len(colors))], ls = lss[(i % len(lss))], marker = markers[(i % len(markers))])
            legends.append('val_task{}'.format(i))
        #plt.legend(legends, loc = 3)
        curr_axis = plt.axis()
        plt.axis([0, curr_axis[1]+1, 0, 1])  
        plt.legend(legends, loc = "upper center", bbox_to_anchor=(0.5, -0.075), ncol = 4)
        #plt.xlabel('epochs')
        plt.xlabel('number of tasks')
        plt.ylabel('validation accuracy')
        plt.savefig(name_rec + "_septasks.png")
        plt.clf()
        
        fig = plt.figure(figsize=(10, 10))
        #plt.title("Averaged performance, " + name_train)
        plt.title("Performance averaged over tasks")
        plt.plot(self.avg_perf[-1], 'r')
        curr_axis = plt.axis()
        plt.axis([curr_axis[0], curr_axis[1]+1, 0, 1])
        for ep_end in self.task_ends[-1]:
            plt.axvline(ep_end, color = 'k', ls = ':')
        plt.xlabel('epochs')
        plt.ylabel('task-averaged validation accuracy ')
        plt.savefig(name_rec + "_avgperf.png")
        plt.clf()
        
        # Finish
        file_rec.close()



        
if __name__ == "__main__":
    
    ## split-MNIST
    
    tasks = list()
    
    mnist = h5py.File('mnist.hdf5', 'r')
    expmng = exp_manager(mnist, num_tasks = 2, num_labels = 10, name_dir = "split_all")
    expmng.split_data(expmng.data, label_per_split = 5, num_labels = 10, resolution = 784)
    
    ### SupGrad_1
    batch_size = 100
    num_epochs = 10 # 400 for bnn, 2000 for nn
    vd_range = 2 # 20 for BNN?
    patience = 1
    init_lr = 1e-4
 
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 1, pri_coeff = 1) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_44_pri{}_{}_{}".format(1, 1, init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    '''
    expmng.init_session()
    model_bnn = bnn_model([784, 100, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 1, pri_coeff = 2) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_1_pri{}_{}_{}".format(1, 2, init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    
    expmng.init_session()
    model_bnn = bnn_model([784, 100, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 1, pri_coeff = 3) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_1_pri{}_{}_{}".format(1, 3, init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    
    expmng.init_session()
    model_bnn = bnn_model([784, 100, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 1, pri_coeff = 4) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_1_pri{}_{}_{}".format(1, 4, init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    '''
    
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 2, pri_coeff = 1) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_44_pri2_1_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')

    
    ### SupGrad_2
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 0, pri_coeff = 1) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_44_pri0_1_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    
    '''
    expmng.init_session()
    model_bnn = bnn_model([784, 100, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 0, pri_coeff = 1) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_1_pri0_1_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    '''
    '''
    ### SupGrad_2
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, kl_reweight = False, train_rho = True, only_loglike = False, ewc = True, squared_std = True, \
                pri_type = 0, pri_coeff = 1) # squared_std?
    expmng.assign_model(model_bnn)
    tasks.append("SpGr_44_pri0_1_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, patience = patience, vdec_range = vd_range, supgrad = True, update_prior = True, reset_q_params = True, kind_prob = 'split')
    '''
    
    batch_size = 100
    num_epochs = 10 # 400 for bnn, 2000 for nn
    vd_range = 2 # 20 for BNN?
    patience = 1
    init_lr = 1e-4
    '''
    ### DNN + EWC
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 100, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = True, l2_reg = False, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("EWC_1_p100_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = True, ewc = True, l2_reg = False, reset_q_params = False, kind_prob = 'split')
    '''
    
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = True, l2_reg = False, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("EWC_44_p100_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = True, ewc = True, l2_reg = False, reset_q_params = False, kind_prob = 'split')
    '''
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 800, 800, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = True, l2_reg = False, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("EWC_88_p100_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = True, ewc = True, l2_reg = False, reset_q_params = False, kind_prob = 'split')
    '''
    
    init_lr = 1e-4
    num_epochs = 10 # 400 for bnn, 2000 for nn
    
    # DNN
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = False, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("DNN_44_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = False, ewc = False, l2_reg = False, reset_q_params = False, kind_prob = 'split')
    
    '''
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 800, 800, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = False, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("DNN_88_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = False, ewc = False, l2_reg = False, reset_q_params = False, kind_prob = 'split')
    '''
    
    # DNN + L2
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = True, reg_penalty = 10)
    expmng.assign_model(model_nn)
    tasks.append("L2_44_p10_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = False, ewc = False, l2_reg = True, reset_q_params = False, kind_prob = 'split')
    
    # DNN + L2-Transfer
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 400, 400, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = True, reg_penalty = 10)
    expmng.assign_model(model_nn)
    tasks.append("L2Tr_44_p10_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = True, ewc = False, l2_reg = True, reset_q_params = False, kind_prob = 'split')
    '''
    # DNN + L2
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 800, 800, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = True, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("L2_88_p100_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = False, ewc = False, l2_reg = True, reset_q_params = False, kind_prob = 'split')
    
    # DNN + L2-Transfer
    expmng.init_session()
    model_nn = nn_shson.nn_model([784, 800, 800, 10], size_data = len(expmng.t_train[0]), size_batch = batch_size, \
                mu = 0.02, outact = tf.nn.relu, seed = 1234, \
                lr = init_lr, ewc = False, l2_reg = True, reg_penalty = 100)
    expmng.assign_model(model_nn)
    tasks.append("L2Tr_88_p100_{}".format(init_lr))
    expmng.train(name_train = tasks[-1], n_epochs = num_epochs, kind_model = "nn", patience = patience, vdec_range = vd_range, supgrad = False, update_prior = True, ewc = False, l2_reg = True, reset_q_params = False, kind_prob = 'split')
    '''
    
    
    '''
    ### OnlineBNN_1
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = 1e-5, kl_reweight = False, train_rho = True, only_loglike = False, ewc = False, squared_std = False) # squared_std?
    expmng.assign_model(model_bnn)
    expmng.train(name_train ="OnlineBNN_784_400_400_10_rstq_1e-5_",n_epochs = num_epochs, patience = 3, vdec_range = vd_range, supgrad = False, update_prior = True, reset_q_params = True)
    
    ### OnlineBNN_2
    expmng.init_session()
    model_bnn = bnn_model([784, 400, 400, 10], size_data = len(expmng.t_train), size_batch = batch_size, \
                mu = 0.02, rhos = [-5.0, 1.0, 10.0], n_samples = 40, outact = tf.nn.relu, seed = 1234, \
                lr = 1e-5, kl_reweight = False, train_rho = True, only_loglike = False, ewc = False, squared_std = False) # squared_std?
    expmng.assign_model(model_bnn)
    expmng.train(name_train ="OnlineBNN_784_400_400_10_nrstq_1e-5_",n_epochs = num_epochs, patience = 3, vdec_range = vd_range, supgrad = False, update_prior = True, reset_q_params = False)
    '''
    
    # Plot
    colors = ['b', 'g', 'r', 'c']
    markers = ['o', '^', 'd', '*', 's']
    lss = ['-', '--', '-.']
    legends = list()
    fig = plt.figure(figsize=(12, 12))
    plt.title("Comparison of task-averaged performances")
    x_plot = range(1, expmng.num_tasks+1)
    for i in range(len(tasks)):
        plt.plot(x_plot, expmng.avg_perf_task[i], colors[(i % len(colors))], ls = lss[(i % len(lss))], marker = markers[(i % len(markers))])
        legends.append(tasks[i])
    
    curr_axis = plt.axis()
    plt.axis([0, curr_axis[1]+1, 0, 1])  
    plt.legend(legends, loc = "upper center", bbox_to_anchor=(0.5, -0.075), ncol = 4)
    #plt.legend(legends, loc = 3)
    plt.xlabel('number of tasks')
    plt.ylabel('averaged performance')
    plt.savefig("{}avgperfs_dnn.png".format(expmng.savedir))
    plt.clf()
    
    
    
    
    
    
    
    

    
    