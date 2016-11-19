import numpy, glob, cv2, csv
import os    
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
K.set_image_dim_ordering('th')

batch_size = 10   #10
nb_classes = 2    #5
nb_epoch = 400   #250
data_augmentation = False#True

#left 17561
#rigth 17562
nb_samples=20 #35123  #20
nb_train_samples=10#28000 #10
nb_test_samples=10 #7123  #10

# input image dimensions
img_rows, img_cols = 256, 256
# the CIFAR10 images are RGB
img_channels = 3

X_data = numpy.zeros((nb_samples, img_channels, img_rows, img_cols), dtype="uint8")
Y_label = numpy.zeros((nb_samples,), dtype="uint8")
aux = []

##############LOADIN DATA##########################################################
print "-----------------------------"
print "Loading dataset..."
j=0
with open('/home/ubuntu-ssd/Documents/INIFIM/classification/data/trainLabels-DL.csv', 'rb') as csvfile:
     trainlabels = csv.reader(csvfile, delimiter=',')
     for row in trainlabels:
      Y_label[j,] = numpy.uint8(row[1])
      aux.append(numpy.str(row[0]))
      j=j+1
      if j==nb_samples:
        break

i=0
for f in aux: #sorted(glob.glob("/home/ubuntu-ssd/Documents/INIFIM/classification/data/256x256/*.jpeg")):
  filename = "256x256__"+f+'.jpeg'
  ruta = '/home/ubuntu-ssd/Documents/INIFIM/classification/data/256x256/'+filename
  img=cv2.imread(ruta) 
  img=img.transpose(2, 0, 1)
  img = img.reshape((1,) + img.shape)
  if i%100==0 :
    print i+1,") ",'\t', f, '\t', Y_label[i]
  X_data[i,:,:,:] = img
  i= i +1
  if i==nb_samples:
  	break

# binary labels (sano: 0 , enfermo: 1)
Y_label_aux = Y_label>0
Y_label = numpy.uint8(Y_label_aux)
print "-----------------------------"
print "Split Data..."
X_train = X_data[0:nb_train_samples]
Y_train = Y_label[0:nb_train_samples]

X_test = X_train
Y_test = Y_train
#X_test = X_data[nb_train_samples:nb_samples]
#Y_test = Y_label[nb_train_samples:nb_samples]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print "-----------------------------"

###################################################################################

# conviertiendo en vector one hot
Y_train = np_utils.to_categorical(Y_train, nb_classes=nb_classes)
Y_test = np_utils.to_categorical(Y_test,nb_classes=nb_classes) 

print X_train.shape[1:]
print Y_train.shape

model = Sequential()

model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
  
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(96, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(96, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(128, 3, 3,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)  #lr=0.005
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())
# Pasando X a float para hacer operaciones 

print 'Float transformation...'
'''
for n in range(0,4):
  X_train[7000*n:7000*(n+1)] = X_train[7000*n:7000*(n+1)].astype('float32')
  print (n+1)*7000 , 'OK'
'''
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print 'Training...'
if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

train_acc = numpy.asarray(hist.history['acc'])
train_loss = numpy.asarray(hist.history['loss'])
test_acc = numpy.asarray(hist.history['val_acc'])
test_loss = numpy.asarray(hist.history['val_loss'])
epocas = []
for i in range(nb_epoch):
  epocas.append(i)

pyplot.figure(figsize=(19,12),dpi=100)
pyplot.suptitle('Train: '+str(nb_train_samples) + '\t' + 'Test: '+str(nb_test_samples), fontsize="x-large")

pyplot.subplot(2,2,1)
pyplot.xlabel('Epoch')
pyplot.ylabel('accuracy')
pyplot.axis([-5, nb_epoch*1.05, 0, 1.1])
pyplot.title('Train acc')
pyplot.plot(epocas,train_acc)

pyplot.subplot(2,2,3)
pyplot.xlabel('Epoch')
pyplot.ylabel('loss')
pyplot.title('Train loss')
pyplot.plot(epocas,train_loss)

pyplot.subplot(2,2,2)
pyplot.xlabel('Epoch')
pyplot.ylabel('accuracy')
pyplot.axis([-5, nb_epoch*1.1, 0, 1.1])
pyplot.title('Test acc')

pyplot.plot(epocas,test_acc)
pyplot.subplot(2,2,4)
pyplot.xlabel('Epoch')
pyplot.ylabel('loss')
pyplot.title('Test loss')
pyplot.plot(epocas,test_loss)
pyplot.savefig('graficas.png')

pyplot.show()


scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy model: %.2f%%" % (scores[1]*100))

###########################################################################################
# saving model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# saving weights
model.save_weights("model.h5")
print("Saved model and weights to disk")
###########################################################################################

###########################################################################################
# re-use model and weoghts
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model and weights from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy loaded model: %.2f%%" % (scores[1]*100))
##########################################################################################
pyplot.figure(figsize=(19,12),dpi=100)
prediction = model.predict(X_test)
pyplot.subplot(2,1,2)
pyplot.xlabel('prediction')
pyplot.ylabel('Sample')
pyplot.grid('on')
pyplot.subplot(2,1,1)
pyplot.xlabel('ground truth')
pyplot.ylabel('Sample')
pyplot.grid('on')

for k in range(nb_test_samples):
  pyplot.subplot(2,1,2)
  pyplot.plot(numpy.asarray(prediction[k][1]), numpy.asarray([k]),'d')
  pyplot.subplot(2,1,1)
  pyplot.plot(Y_test[k,1], numpy.asarray([k]),'d')

pyplot.axis([-0.1,1.1, -1,nb_test_samples+1 ])
pyplot.savefig('prediction.png')
pyplot.show()
#print prediction
'''
aux_train = aux[0:nb_train_samples]
aux_test = aux[nb_train_samples:nb_samples]
#aux_test = aux_train

num_img=5
# probando con una imagen
pyplot.figure('Test - one image')
pyplot.imshow(toimage(X_test[num_img-1]))
pyplot.title(aux_test[num_img-1])
pyplot.show()
print 'label:', Y_test[num_img-1]
x = numpy.expand_dims(X_test[num_img-1], axis=0)
prediction = model.predict(x)
print 'prediction:', numpy.round(prediction)
'''

'''
print "-----------------------------"
print "Normalization: Zero Mean , standard deviation 1 , min-max 0-1"

mean_train = X_train.mean(0)
deviation = X_train.std(0)
X_train=(X_train - mean_train)/deviation
X_train = ( X_train - X_train.min(0) )/(X_train.max(0)-X_train.min(0))

print "-----------------------------"
'''
