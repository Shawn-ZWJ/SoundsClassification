import mxnet as mx
from mxnet import gluon, nd, autograd

def get_model(num_labels):
    model = gluon.nn.Sequential()

    with model.name_scope():
        model.add(gluon.nn.Dense(256, in_units=40,activation="relu")) # 1st layer (256 nodes)

        model.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer

        model.add(gluon.nn.Dense(num_labels))
    model.collect_params().initialize(mx.init.Normal(1.))
    return model