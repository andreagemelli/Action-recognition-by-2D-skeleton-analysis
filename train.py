from keras.callbacks import TensorBoard
from keras import optimizers
from utils import *
from model import *
from math import inf
from time import time
import numpy as np
import tensorflow as tf
import random


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

# dimension of reshape, dist_norm for distance normalization and cos_norm for cosine normalization
reshape_dist_norm = [-1, 15, 2]
reshape_cos_norm = [-1, 8, 1]

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
dataset_path = 'dataset'
weights_path = 'weights/model.h5'
num_classes = 6

dataset = Action_Dataset(dataset_path)
training, test = dataset.get_data(10)
split = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
best_loss = inf

for t in range(0, len(training)):
    # hyperparams
    lr = 0.001
    epochs = 200

    # model
    model = one_obj()
    model.compile(optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # train / val split
    train = {}
    val = training[t]

    for e in range(epochs):
        X_0 = []
        X_1 = []
        Y = []

        V_0 = []
        V_1 = []
        YV = []

        for i in range(1, num_classes + 1):  # loop 6 classes
            train[i] = []
            for s in split:
                if s != t:
                    train[i].extend(training[s][i])

            # generate training data
            for j in range(len(train[i])):  # loop all samples within the same class
                # First person pose
                p_0 = np.copy(normalize_by_center(train[i][j])) # function of normalization
                p_0 = p_0.reshape(reshape_dist_norm) # reshape dimension
                t_0 = p_0.shape[0]  # the number of all frames
                if t_0 > 16:  # sample the range from crop size of [16,t_0]
                    ratio = np.random.uniform(1, t_0 / 16)
                    l = int(16 * ratio)
                    start = random.sample(range(t_0 - l), 1)[0]
                    end = start + l
                    p_0 = p_0[start:end, :, :]
                    p_0 = zoom(p_0)
                elif 0 < t_0 <= 16:
                    p_0 = zoom(p_0)

                if p_0.size != 0:
                    # Calculate the temporal difference
                    a = p_0[1:, :, :]
                    b = p_0[:-1, :, :]
                    p_0_diff = p_0[1:, :, :] - p_0[:-1, :, :]
                    c = p_0_diff[-1, :, :]
                    p_0_diff = np.concatenate((p_0_diff, np.expand_dims(p_0_diff[-1, :, :], axis=0)))

                    X_0.append(p_0)
                    X_1.append(p_0_diff)

                    label = np.zeros(num_classes)
                    label[i - 1] = 1
                    Y.append(label)

            # generate validation data
            for j in range(len(val[i])):  # loop all samples within the same class
                # First person pose
                p_0 = np.copy(normalize_by_center(val[i][j]))  # function of normalization
                p_0 = p_0.reshape(reshape_dist_norm)  # reshape dimension
                t_0 = p_0.shape[0]  # the number of all frames
                if t_0 > 16:  # sample the range from crop size of [16,t_0]
                    ratio = np.random.uniform(1, t_0 / 16)
                    l = int(16 * ratio)
                    start = random.sample(range(t_0 - l), 1)[0]
                    end = start + l
                    p_0 = p_0[start:end, :, :]
                    p_0 = zoom(p_0)
                elif 0 < t_0 <= 16:
                    p_0 = zoom(p_0)

                if p_0.size != 0:
                    # Calculate the temporal difference
                    a = p_0[1:, :, :]
                    b = p_0[:-1, :, :]
                    p_0_diff = p_0[1:, :, :] - p_0[:-1, :, :]
                    c = p_0_diff[-1, :, :]
                    p_0_diff = np.concatenate((p_0_diff, np.expand_dims(p_0_diff[-1, :, :], axis=0)))

                    V_0.append(p_0)
                    V_1.append(p_0_diff)

                    label = np.zeros(num_classes)
                    label[i - 1] = 1
                    YV.append(label)

        X_0 = np.stack(X_0)
        X_1 = np.stack(X_1)
        Y = np.stack(Y)

        V_0 = np.stack(V_0)
        V_1 = np.stack(V_1)
        YV = np.stack(YV)

        history = model.fit([X_0, X_1], Y, batch_size=32, epochs=1, verbose=1, shuffle=True, validation_data=([V_0, V_1], YV), callbacks=[TrainValTensorBoard(write_graph=False)])

        print("")
        print('SPLIT', (t + 1), '/', len(split), '- EPOCHS', (e + 1), '/', epochs)

        if not (e + 1) % 10:
            if float(history.history['val_loss'][0]) < best_loss:
                print("Val loss improoved from", best_loss, "to", history.history['val_loss'][0])
                best_loss = history.history['val_loss'][0]
                model.save_weights(weights_path)
                model.load_weights(weights_path)
            else:

                if lr >= 1e-6:
                    lr *= 0.1
                    print("New learning rate:", lr)
                    adam = optimizers.Adam(lr)
                    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
                print("Val loss did not improoved from", best_loss)
