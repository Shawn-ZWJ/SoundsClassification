from mxnet.gluon.utils import download


import time
import numpy as np
import logging
import mxnet as mx
from mxnet import gluon, nd, autograd
from soundsclassification import SoundsClassification

def main():
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized!")

    # Grab the dataset
    print("Grabbing the dataset...")
    tick = time.time()
    sdclassx = _get_data()
    tock=time.time()

    print("The time taken to download the dataset and populate the data with pre processing :  ",(tock-tick)," seconds")

    num_op_labels = len(sdclassx._le.classes_)

    #Creating model
    model = get_model(num_op_labels)

    #Getting the loss function
    loss = get_loss_function()

    # Getting the trainer
    trainer = get_trainer(model)

    epochs = 30
    batch_size = 32
    tick = time.time()
    train(sdclassx, model, trainer, loss, epochs=epochs, batch_size=batch_size)
    tock = time.time()
    print("Training the sound classification for ",epochs," epochs, MLP model took ",(tock-tick)," seconds")





def _get_data():
    
    sdclassx = SoundsClassification(transform=None)
    return sdclassx
    
    

def get_model(num_labels):
    model = gluon.nn.Sequential()

    with model.name_scope():
        model.add(gluon.nn.Dense(256, in_units=40,activation="relu")) # 1st layer (256 nodes)

        model.add(gluon.nn.Dense(256, activation="relu")) # 2nd hidden layer

        model.add(gluon.nn.Dense(num_labels))
    model.collect_params().initialize(mx.init.Normal(1.))
    return model


def get_trainer(model):
    return gluon.Trainer(model.collect_params(), 'adadelta')


def get_loss_function():
    return gluon.loss.SoftmaxCELoss(from_logits=False, sparse_label=True)


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):        
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)

    return acc.get()[1]



def train(sdclassx, model, trainer, loss_fn, epochs=30, batch_size=32):
    num_examples = 5435
    cumulative_losses = []
    # training_accuracies = []
    audio_train_loader = gluon.data.DataLoader(sdclassx, shuffle=True, batch_size=32)
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(audio_train_loader):

            with autograd.record():
                output = model(data)

                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += mx.nd.sum(loss).asscalar()
        cumulative_losses.append(cumulative_loss/num_examples)
        if e%5==0:
            train_accuracy = evaluate_accuracy(audio_train_loader, model)
            print("Epoch %s. Loss: %s Train accuracy : %s " % (e, cumulative_loss/num_examples, train_accuracy))

    train_accuracy = evaluate_accuracy(audio_train_loader, model)
   
    print("Final training accuracy: ",train_accuracy)

   


    print("Training complete....")
if __name__ == '__main__':
    main()