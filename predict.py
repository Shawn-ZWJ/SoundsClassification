import mxnet as mx
from mxnet import gluon, nd, autograd
import os
import librosa
import numpy as np

def get_model(num_labels):
    model = gluon.nn.Sequential()

    with model.name_scope():
        model.add(gluon.nn.Dense(256,activation="relu")) # 1st layer (256 nodes)

        model.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer

        model.add(gluon.nn.Dense(num_labels))
    model.collect_params().initialize(mx.init.Normal(1.))
    return model

def load_model_params():
    curr_dir = os.getcwd()
    if not os.path.exists(os.path.join(curr_dir,'models')):
        raise RuntimeError("Trying to predict without a model... Exitting. Try saving the model whemn you train the next time!")
    
    model_dir = os.path.join(curr_dir, "models")

    model = get_model(10)
    
    model.load_parameters(os.path.join(model_dir,'model.params'), ctx=mx.cpu())

    return model

### For reference..... ####
"""
           'air_conditioner' :      0,
            'car_horn':             1,
            'children_playing' :    2,
            'dog_bark':             3,
            'drilling':             4, 
            'engine_idling':        5, 
            'gun_shot' :            6, 
            'jackhammer':           7, 
            'siren':                8,
            'street_music':         9
     
"""


def main(**kwargs):
    model = load_model_params()
    filename = "./Train/10"

    # if not os.path.exists(filename):
    #     raise RuntimeError("Filename Path: ", filename, " does not exist! Exitting!")

    # Extracting features from the filename trying to load...
    X1, sample_rate = librosa.load(str(filename)+".wav", res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X1, sr=sample_rate, n_mfcc=40).T,axis=0)
    mfccs = mfccs.reshape((1,-1))
    print("Features shape: ",mfccs.shape)
    prediction = nd.argmax(model(nd.array(mfccs)), axis=1)
    print("Predicted for the wav file: ")
    print(prediction)



if __name__ == '__main__':
    filename = "./Train/23.wav"
    main(filename=filename)