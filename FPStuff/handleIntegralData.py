import numpy as np
import h5py

def retrieveData(fileName):

    dir = "savedData/"

    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'r')
    cutOffs = h5f['cutoff'][:]
    dosInt = h5f['dosInt'][:]
    h5f.close()

    return (cutOffs, dosInt)

def writeData(cutoff, dosInt, fileName):

    dir = "savedData/"
    fileName = dir + fileName + ".h5"

    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('cutoff', data=cutoff)
    h5f.create_dataset('dosInt', data=dosInt)
    h5f.close()