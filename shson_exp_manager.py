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


def shuffle_data(data):
    dlen = range(len(data))
    random.shuffle(dlen)
        
    return np.array(data)[dlen]
    
    