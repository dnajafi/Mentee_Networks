# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 20:20:06 2016

@author: GHANSHYAM
"""
import os
import scipy.io
import numpy
import gzip
import cPickle as pickle
datasets=[['mnist_background_images\\mnist_background_images_test.amat','mnist_background_images\\mnist_background_images_train.amat'],['mnist_background_random\\mnist_background_random_test.amat','mnist_background_random\\mnist_background_random_train.amat']]
train_set=()
valid_set=()
test_set=()

inputmatrix=[]
targetlist=[]
inputmatrixtest=[]
targetlisttest=[]
'''
for line in open(os.path.join('data',datasets[0][0])):
    mylist=line.split()
    #x.append(mylist[784])
    x=mylist[0:784]
    inputmatrix.append(x)
    targetlist.append(mylist[784])
input=numpy.array(inputmatrix)
target=numpy.array(targetlist)
train_set=(input,target)
print (numpy.array(inputmatrix[:2000]).shape)
print (train_set)
print ('matrix')
print (input.shape)
print ('vector')
print (target.shape)
'''


def load_mnist_variants(d):
    #print (os.listdir('data'))
    for name in os.listdir(d):
        print ('loading from ')
        print (name)
        
        if "train" in name:

            for line in open(os.path.join(d,name)):
                
                mylist=line.split()
                x=mylist[0:784]
                inputmatrix.append(x)
                targetlist.append(mylist[784])
            input=numpy.array(inputmatrix)
            target=numpy.array(targetlist)
            train_set=(input,target)
            valid_set=(numpy.array(inputmatrix[:2000]),numpy.array(targetlist[:2000]))
            
        if "test" in name:
            for line in open(os.path.join(d,name)):
                mylist=line.split()
                x=mylist[0:784]
                inputmatrixtest.append(x)
                targetlisttest.append(mylist[784])
            test_set=(numpy.array(inputmatrixtest),numpy.array(targetlisttest))
    #return (train_set,valid_set,test_set)
    with gzip.open('mnist_rotation_new.pkl.gz','wb') as fileobj:
        pickle.dump((train_set,valid_set,test_set),fileobj)

load_mnist_variants('data\\mnist_rotation_new\\')