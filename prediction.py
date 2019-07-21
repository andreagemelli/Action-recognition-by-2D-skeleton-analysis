from keras import optimizers
from utils import *
from model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# path to be modified
json_folder = 'example/handwaving/jsons'
weights_path = 'weights/model_distance_center.h5'

list_class = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
test_data = Action_Dataset('')  # here used just to call the read_json function, no matter the input path

# loading model
lr=0.001
adam = optimizers.Adam(lr)
model = one_obj()
model.compile(adam, loss='categorical_crossentropy')
model.load_weights(weights_path)

X_0 = []
X_1 = []

# label vector
Y = [0, 0, 0, 0, 0, 0]
action_name = json_folder.split('/')[-2]
for i, j in enumerate(list_class):
    if action_name == j:
        Y[i] = 1

p_0 = [test_data.read_json(os.path.join(json_folder, json)) for json in os.listdir(json_folder)]
for i in range(len(p_0), 0, -1):
    if not p_0[i-1]:
        del p_0[i-1]

p_0 = normalize_by_center(p_0)
p_0 = np.array(p_0)
p_0 = p_0.reshape([-1, 15, 2])
t_0 = p_0.shape[0]  # the number of all frames

start = 0
l = 16

for i in range(0, int(t_0 / 16) + 1):

    end = start + l

    if end > t_0:
        end = t_0

    p = p_0[start:end, :, :]
    p = zoom(p)

    # Calculate the temporal difference
    p_diff = p[1:, :, :] - p[:-1, :, :]
    p_diff = np.concatenate((p_diff, np.expand_dims(p_diff[-1, :, :], axis=0)))

    X_0.append(p)
    X_1.append(p_diff)

    start = end

X_0 = np.stack(X_0)
X_1 = np.stack(X_1)
Y = np.stack(Y)

history = model.predict([X_0, X_1], batch_size=1, verbose=0)

if np.argmax(Y) == np.argmax(history[0]):
    print("Correct prediction!", list_class[np.argmax(history[0])])
else:
    print("Wrong prediction!")
    print('Expected:', list_class[np.argmax(Y)], '- Resulted:', list_class[np.argmax(history[0])])
