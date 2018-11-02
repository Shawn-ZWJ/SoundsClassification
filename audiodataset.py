from mxnet.gluon.utils import download

import os
import librosa
import time
import numpy as np

import mxnet as mx
from mxnet import gluon, nd, autograd
import zipfile
from sklearn.preprocessing import LabelEncoder


class AudioDataset(gluon.data.dataset._DownloadedDataset):
    """Generic audio dataset 

    Each sample is an audio clip with a sound from the street .

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/yesno
        Path to temp folder for storing data (waves_yesno.tar.gz)
       
    """
    def __init__(self, url=None, root='.',
                 train=True, transform=None):
        
        self._namespace = 'yesno'
        self._url = url # url = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'
        self._download_dir= "."
        print("In audio dataset passing root as ",root)
        super(AudioDataset, self).__init__(root=root, transform=transform)
        

    def _get_data(self):
        raise NotImplementedError(" You need to implement this method in the child class!")
    

    def _preprocess(self, temp_data, sampling_rate  = 22500, method = 'mfcc', n_fcc=40):
        """
            For each file in temp_data, extract mel spectrogram features and populate the _data
        """
        print("Preprocessing the data...")
        tick = time.time()
        self._data = []
        for record in temp_data:
           
           if method=='mfcc':
                #print("In _preprocess of audio dataset...")
                mfccs = np.mean(librosa.feature.mfcc(y=record, sr=sampling_rate, n_mfcc=n_fcc).T,axis=0)
                self._data.append(mfccs)
        self._data = nd.array(self._data)
        tock = time.time()
        print("Pre processing done!")
        print("AudioDataset: Preprocessing completed in ",(tock-tick)," seconds.")