import tensorflow as tf
import keras
from keras import applications, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras import optimizers
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Data directory
DATA_DIR = './'
X_TRAIN_PATH = './x_train.npy'
Y_TRAIN_PATH = './y_train.npy'
MODEL_PATH = 'trained_model.h5'

# SPARK parameters
NUM_PARTITION = 5


img_width, img_height = 224,224

x_train = []
y_train = []
'''
for landmark_class in sorted(os.listdir(DATA_DIR)):
    if landmark_class.startswith('.'):
      continue
    label = landmark_class
    #print(label)
    # read image
    for im in os.listdir(os.path.join(DATA_DIR, landmark_class)):
        if im.startswith('.'):
          continue
        #print(im)
        try:
          img = Image.open(os.path.join(DATA_DIR, landmark_class, im)) # open the image file
          img.verify()
          image = skimage.io.imread(os.path.join(DATA_DIR, landmark_class, im))
          if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
          #print(image.shape)
          image = skimage.transform.resize(image,(img_height,img_width,3), preserve_range=True)
          x_train.append(image)
          y_train.append(label)
        except (IOError, SyntaxError) as e:
          print('Bad file:', label, im)

x_train = np.array(x_train)
x_train = x_train.astype('float16')
# normalize to the range 0-1
x_train /= 255.0

y_train = np.array(y_train).reshape(-1, 1)
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)
# One-hot encoding on labels
y_train = y_train.reshape(-1, 1)
onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()
'''

x_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)
print(">> train data loaded")
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
# Shuffle the order?
x_train, y_train = shuffle(x_train, y_train)
print(">> train data shuffled")
# create keras model

# VGG
base_model = applications.VGG16(weights='imagenet',include_top= False,input_shape=(224,224,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dense(256, activation='relu'))
n_class = 1066
top_model.add(Dense(n_class, activation='softmax'))

model = Model(input= base_model.input, output= top_model(base_model.output))
print(len(model.layers))
# set the first 16 layers to non-trainable (weights will not be updated) - 1 conv layer and three dense layers will be trained
for layer in model.layers[:16]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-8, decay=0.0),metrics=['accuracy'])
print(">> model compiled")
# Train model
overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 5, verbose=1)
history = model.fit(x_train, y_train, batch_size=32, verbose=1, validation_split=0.2,
          epochs=10000000, callbacks=[overfitCallback])
print(">> model trained")
# save model
model.save(MODEL_PATH)
print(">> model saved")

# Plot training & validation accuracy values
plt.plot(history.history['acc'], color='blue')
plt.plot(history.history['val_acc'], color='green')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_curve.png')

# Plot training & validation loss values
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'], color='green')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_curve.png')
# trained_model = keras.models.load_model('path_to_my_model.h5')
# predictions = trained_model.predict(x_test)