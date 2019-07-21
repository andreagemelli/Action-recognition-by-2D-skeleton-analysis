import matplotlib.pyplot as plt
import random

from keras import optimizers
from utils import *
from model import *

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# path to test data
KTH_ACTION = 'dataset'
dataset = Action_Dataset(KTH_ACTION)
train, test = dataset.get_data(10)

list_class = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 0.001
adam = optimizers.Adam(lr)
model = one_obj()
model.compile(adam, loss='categorical_crossentropy')
model.load_weights('weights/model_distance_center.h5')
model.summary()

X_0 = []
X_1 = []
Y = []

for i in range(1, 7):  # loop 6 classes
    for j in range(len(test[i])):  # loop all samples within the same class

        p_0 = np.copy(np.array(test[i][j]))
        p_0 = np.array(normalize_by_center(p_0))
        p_0 = p_0.reshape([-1, 15, 2])
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
            # Calculate the temporal difference
        if p_0.size != 0:
            p_0_diff = p_0[1:, :, :] - p_0[:-1, :, :]
            p_0_diff = np.concatenate((p_0_diff, np.expand_dims(p_0_diff[-1, :, :], axis=0)))

            X_0.append(p_0)
            X_1.append(p_0_diff)

            label = np.zeros(6)
            label[i - 1] = 1
            Y.append(label)

X_0 = np.stack(X_0)
X_1 = np.stack(X_1)
Y = np.stack(Y)
#print(Y)

history = model.predict([X_0, X_1], batch_size=1, verbose=1)
print("Test example:", Y.shape[0])
count = [0, 0, 0, 0, 0, 0]
classes = [0, 0, 0, 0, 0, 0]

y_test = []
y_pred = []

for i in range(0, Y.shape[0]):
    y_test.append(np.argmax(Y[i]))
    y_pred.append(np.argmax(history[i]))
    if y_test[i] == y_pred[i]:
        count[y_pred[i]] += 1
    classes[y_test[i]] += 1

print("Predictions:")
for i in range(0, len(classes)):
    print(list_class[i], count[i], "/", classes[i])
print("Total", np.sum(count), "/", np.sum(classes),  "-", str((np.sum(count) / np.sum(classes))*100), "%")

plot_confusion_matrix(np.array(y_test), np.array(y_pred), classes=np.array(list_class), normalize=True,
                      title='Normalized confusion matrix')

plt.show()