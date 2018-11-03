import mxnet as mx
from mxnet import gluon, nd, autograd
import os
import librosa
import numpy as np

def get_model(num_labels):
    model = gluon.nn.Sequential()

    with model.name_scope():
        model.add(gluon.nn.Dense(256, in_units=40,activation="relu")) # 1st layer (256 nodes)

        model.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer

        model.add(gluon.nn.Dense(num_labels))
    model.collect_params().initialize(mx.init.Normal(1.))
    return model

def load_params(num_labels = 10):
    models_dir = os.path.join(os.getcwd(),'models')
    model_param_file = os.path.join(models_dir, 'model.params')
    
    model = get_model(num_labels)
    model.load_params(model_param_file)
    return model



"""
    for reference:

       'air_conditioner':   0,
       'car_horn':          1,
       'children_playing':  2,
       'dog_bark':          3,
       'drilling':          4,
       'engine_idling':     5,
       'gun_shot':          6,
       'jackhammer':        7,
       'siren':             8,
       'street_music':      9
"""
def main():
    filename = "1.wav"
    filename = os.path.join(os.getcwd(), "Train", filename)
    
    model  = load_params(10)


    X1, sampling_rate = librosa.load(str(filename), res_type='kaiser_fast')
    feature_vec = np.mean(librosa.feature.mfcc(y=X1, sr=sampling_rate, n_mfcc=40).T,axis=0)
    output = model(nd.array(feature_vec).reshape((1,-1)))
    prediction = nd.argmax(output,axis = 1)
    print("For File: ",filename," the predicted label is: ")
    int_label = (int)(prediction.asscalar())
    print( int_label)


if __name__=='__main__':
    main()