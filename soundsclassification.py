from mxnet.gluon.utils import download

import os
import librosa
import time
import numpy as np

import mxnet as mx
from mxnet import gluon, nd, autograd
import zipfile
from sklearn.preprocessing import LabelEncoder


class SoundsClassification(gluon.data.dataset._DownloadedDataset):
    """Sounds dataset from http://www.openslr.org/resources/1/waves_yesno.tar.gz

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
        
        super(SoundsClassification, self).__init__(root, transform)
        

    def _get_data(self):
        self._data=[]
        self._label=[]
        #Downloading the audio dataset from the url
        source_tar_ns = 'yesnogz/'
#         gz_dir = download(self._url,overwrite=True, path=source_tar_ns)
        
        zip_ref = zipfile.ZipFile('./train.zip', 'r')
        zip_ref.extractall(".")
        zip_ref.close()
        
        
        # Read the file fro the label and populate X and y
        train_csv = './train.csv'
        train_dir = './Train'
        
        i=0
        with open(train_csv, "r") as traincsv:
            for line in traincsv:
                
                self._data.append(os.path.join(train_dir,line.split(",")[0]))
                self._label.append(line.split(",")[1].strip())
        
        self._data = self._data[1:]
        
        #Label encoding being done
        self._le = LabelEncoder()
        self._label = self._label[1:]
        self._label = np.array(self._le.fit_transform(self._label))
        
        temp_data = self._data[:]
        self._preprocess(temp_data)
    
    def _preprocess(self, temp_data):
        """
            For each file in temp_data, extract mel spectrogram features and populate the _data
        """
        print("Preprocessing the data...")
        self._data = []
        for flnm in temp_data:
            X1, sample_rate = librosa.load(str(flnm)+".wav", res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X1, sr=sample_rate, n_mfcc=40).T,axis=0)
            self._data.append(mfccs)
        self._data = nd.array(self._data)
        print("Pre processing done!")