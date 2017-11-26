import os, sys, time, datetime, random
import numpy as np

def get_datetime():
    return datetime.datetime.now().isoformat()

def get_timedir():
    tdir = get_datetime()
    tdir1 = tdir.split(":")
    tdir2 = tdir1[0].split("-")
    tdir3 = tdir1[-1].split(".")
    
    resdir = tdir2[0] + tdir2[1] + tdir2[2] + tdir1[1] + tdir3[0] + tdir3[1][:2]
    
    return resdir

def record_time():
    return time.time()
    
def elapsed_time(past):
    return time.time() - past


'''
the form of 'path' should be 'PATHNAME/'.
'''
def make_savedir(path = ""):
    savedir = "./" + path + get_timedir() + "/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    return savedir


def shuffle_data(x_data, t_data, kind_prob = 'permute'):
    if kind_prob == 'permute': 
        dlen = range(len(t_data))
        random.shuffle(dlen)
        res_x = list()
        for xd in x_data:
            res_x.append(np.array(xd)[dlen])
            
        return res_x, np.array(t_data)[dlen]
    
    elif kind_prob == 'split': 
        res_x = list()
        res_t = list()
        for i in range(len(x_data)):
            dlen = range(len(t_data[i]))
            random.shuffle(dlen)
            res_x.append(np.array(x_data[i])[dlen])
            res_t.append(np.array(t_data[i])[dlen])
        
        return res_x, res_t
    
    