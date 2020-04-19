import tensorflow as tf
import keras
from keras import applications, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
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

# DL parameters
NB_CLASSES = 1066
nn1 = 1024; nn2 = 1024; nn3 = 200
lr = 0.0001; decay=0
batch_size = 1024
dropout = 0.5
l1 = 0.0001
l2 = 0.0001

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
base = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
for layer in base.layers:
    layer.trainable = False
opt = keras.optimizers.Adam(lr=lr)
reg = keras.regularizers.l1_l2(l1=l1, l2=l2)
x = Flatten()(base.output)
x = Dense(nn1, activation='relu', kernel_regularizer=reg)(x)
x = BatchNormalization()(x)
x = Dense(nn2, activation='relu', kernel_regularizer=reg)(x)
x = BatchNormalization()(x)
x = Dense(nn3, activation='relu', kernel_regularizer=reg)(x)
x = BatchNormalization()(x)
x = Dense(NB_CLASSES, activation='softmax')(x)
model = Model(input=base.input, output=x)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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