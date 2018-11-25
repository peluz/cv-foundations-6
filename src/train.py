import numpy as np
from utils import *
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras import backend as K
import os


DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../data/lfw-deepfunneled")


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_base_network():
    inp = Input(shape=(2048,))
    x = Dense(1024, activation="relu")(inp)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    return Model(inp, x)


def compute_accuracy(y_true, y_pred, threshold=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


people_dict = load_people_dict()
X_train, y_train = load_data(os.path.join(DATA_PATH, "../pairsDevTrain.txt"), people_dict)
X_val, y_val = load_data(os.path.join(DATA_PATH, "../pairsDevTest.txt"), people_dict)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

base_network = create_base_network()
input_a = Input(shape=(2048,))
input_b = Input(shape=(2048,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
tensorboard = TensorBoard(log_dir=os.path.join(DIRNAME, "../models/baseline"),
                          batch_size=128, update_freq="batch")
saver = ModelCheckpoint(os.path.join(DIRNAME, "../models/baseline") + "/model.hdf5", 
                        verbose=1,
                        save_best_only=True, monitor="val_loss",
                        mode="min")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5,
                              patience=5, verbose=1, min_lr=0.0001)
model.fit([X_train[:, 0], X_train[:, 1]], y_train,
          batch_size=128,
          epochs=100,
          validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
          callbacks=[saver, reduce_lr, tensorboard],
          verbose=2)

# compute final accuracy on training and test sets
y_pred = model.predict([X_train[:, 0], X_train[:, 1]])
print(y_pred)
train_acc = compute_accuracy(y_train, y_pred)
y_pred = model.predict([X_val[:, 0], X_val[:, 1]])
val_acc = compute_accuracy(y_val, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * val_acc))
print(model.predict([people_dict["Aaron_Peirsol"][1], people_dict["Aaron_Peirsol"][2]]))
print(model.predict([people_dict["Aaron_Peirsol"][1], people_dict["Nicole_Kidman"][0]]))
print(model.predict([people_dict["Nicole_Kidman"][1], people_dict["Nicole_Kidman"][0]]))
