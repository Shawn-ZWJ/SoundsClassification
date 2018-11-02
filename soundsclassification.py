from mxnet.gluon.utils import download

import os
import librosa
import time
import numpy as np

import mxnet as mx
from mxnet import gluon, nd, autograd
import zipfile
from sklearn.preprocessing import LabelEncoder
from audiodataset import AudioDataset


class SoundsClassification(AudioDataset):
    """Sounds dataset from Sounds classification example

    Each sample is an audio clip with a sound from the street .

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/yesno
        Path to temp folder for storing data (waves_yesno.tar.gz)
       
    """
    def __init__(self, url=None, root=str(os.getcwd()),
                 train=True, transform=None):
        
        self._namespace = 'yesno'
        self._url = url # url = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'
        self._download_dir= "."
        
        super(SoundsClassification, self).__init__(root = root, transform = transform)
        

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
        
       
        temp_data = self._data[1:]
        
        #Label encoding being done
        self._le = LabelEncoder()
        self._label = self._label[1:]
        self._label = np.array(self._le.fit_transform(self._label))
        self._sr = 22500
        self._raw_data = []
        for flnm in temp_data:
            X1, sample_rate = librosa.load(str(flnm)+".wav", res_type='kaiser_fast')
            self._raw_data.append(X1)

        
        self._preprocess(method='mfcc', sampling_rate = self._sr)
    

    def _preprocess(self, method='mfcc', sampling_rate = 22050, n_fcc = 40 ):
        """
            For each file in temp_data, extract mel spectrogram features and populate the _data
        """
        print("Pre processing in the Sound class")
        super(SoundsClassification, self)._preprocess(self._raw_data, method=method, sampling_rate=sampling_rate, n_fcc=n_fcc)
   