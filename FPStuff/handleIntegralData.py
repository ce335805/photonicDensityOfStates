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

def retrieveDataFixedCutoff(fileName):

    dir = "savedData/"

    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    dArr = h5f['distance'][:]
    dosInt = h5f['dosInt'][:]
    h5f.close()

    return (cutoff, dArr, dosInt)

def writeDataFixedCutoff(cutoff, dArr, dosInt, fileName):

    dir = "savedData/"
    fileName = dir + fileName + ".h5"

    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('cutoff', data=np.array([cutoff]))
    h5f.create_dataset('distance', data=dArr)
    h5f.create_dataset('dosInt', data=dosInt)
    h5f.close()


def retrieveMassData(fileName):
    dir = "savedData/"
    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    dArr = h5f['distance'][:]
    massArr = h5f['massArr'][:]
    h5f.close()

    return (cutoff[0], dArr, massArr)

def writeMassData(cutoff, dArr, massArr, fileName):
    dir = "savedData/"
    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('cutoff', data=np.array([cutoff]))
    h5f.create_dataset('distance', data=dArr)
    h5f.create_dataset('massArr', data=massArr)
    h5f.close()

def readMassesSPhP(filename):

    dir = "../SphPStuff/savedData/"
    fileName = dir + filename + ".hdf5"

    h5f = h5py.File(fileName, 'r')
    cutoff = h5f['cutoff'][:]
    mArr = h5f['delM'][:]
    zArr = h5f['zArr'][:]
    h5f.close()
    return (cutoff, zArr, mArr)

def writeDosPaper(d, wArr, dos):

    dir = "../SphPStuff/savedData/PaperData/"
    fileName = "DosFP"
    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('d', data=np.array([d]))
    h5f.create_dataset('wArr', data=wArr)
    h5f.create_dataset('dos', data=dos)
    h5f.close()


def writeDosDefence(d, wArr, dos):

    dir = "../SphPStuff/savedData/DefenceData/"
    fileName = "DosFP"
    fileName = dir + fileName + ".h5"
    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('d', data=np.array([d]))
    h5f.create_dataset('wArr', data=wArr)
    h5f.create_dataset('dos', data=dos)
    h5f.close()
