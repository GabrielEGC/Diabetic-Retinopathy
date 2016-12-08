import numpy, glob, cv2, csv, sys
import os    
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"

# fix random seed for reproducibility
numpy.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint
K.set_image_dim_ordering('th')

lr = 0.02 
# 0.03 decay 1e-4
 
batch_size = 128#128   #10
nb_classes = 2    #5
nb_epoch = 100 #200#400   #25
data_augmentation = False #Ver seccin de data augmentation para activar opciones (line 206 aprox)

side = "left" #right
etiquetas = 17562 #numleft = 17562 #numrigth = 17562        <------> labels
nb_train_samples= 15000#4000*2
nb_test_samples= 2562#692*2
nb_samples=nb_train_samples + nb_test_samples
same=0 #Flag Xtrain = Xtest   otherwise 0 =->Xtrain and test different

#Data por clases   ----> 
'''
left (de 1 a 2.74)
0: 12870  1: 4692

Right (de 1 a 2.79)
0: 12938  1:4624
'''
data_per_classes =0 #flag activar distribcion 50/50
num_zero_class= nb_train_samples/2
num_one_class = nb_train_samples/2

# input image dimensions
img_rows, img_cols = 256, 256
# the CIFAR10 images are RGB
img_channels = 3

X_data = numpy.zeros((nb_samples, img_channels, img_rows, img_cols), dtype="uint8")
Y_label = numpy.zeros((etiquetas,), dtype="uint8")

##############LOADIN DATA##########################################################
print "-----------------------------"
print "Loading dataset..."

# Loading .csv -> etiquetas
aux = []
j=0
if side == "left":
  ruta_csv = '/home/ubuntu-ssd/Documents/INIFIM/Data/preprocesada/trainLabels-DL.csv'
elif side == "right":
  ruta_csv = '/home/ubuntu-ssd/Documents/INIFIM/Data/preprocesada/trainLabels-DR.csv'
with open(ruta_csv, 'rb') as csvfile:
     trainlabels = csv.reader(csvfile, delimiter=',')
     for row in trainlabels:
      Y_label[j,] = numpy.uint8(row[1])
      aux.append(numpy.str(row[0]))
      j=j+1

aux = numpy.asarray(aux)
# binary labels (sano: 0 , enfermo: 1)
Y_label_aux = Y_label>0
Y_label = numpy.uint8(Y_label_aux)


if data_per_classes ==1:
  if num_zero_class+num_one_class==nb_train_samples:
    var = numpy.asarray([aux,Y_label])
    var = var.T
    #numpy.random.shuffle(var)
    data_0 = var[Y_label==0]
    data_1 = var[Y_label>0]
    #numpy.random.shuffle(data_0)
    #numpy.random.shuffle(data_1)
    data_t_0 = data_0[0:num_zero_class]
    data_t_1 = data_1[0:num_one_class]
    data_te_0 = data_0[num_zero_class:(num_zero_class+(nb_test_samples/2))]
    data_te_1 = data_1[num_one_class:(num_one_class+(nb_test_samples/2))]
    data = numpy.concatenate((data_t_0,data_t_1,data_te_0,data_te_1),axis=0)
    #numpy.random.shuffle(data)                          # desordenar data
    aux = data[:,0]                                 # Nombre de la imagen
    Y_label = numpy.uint8(data[:,1])                      # etiquetas
  else:
    sys.exit("Error - (num_zero_class + num_one_class) diferente de nb_train_samples -- Line 35 - 47")

if img_rows == 256:
  ruta_img = '/home/ubuntu-ssd/Documents/INIFIM/Data/preprocesada/substract_local_mean_color/256x256/256x256__'
elif img_rows == 512:
  ruta_img = '/home/ubuntu-ssd/Documents/INIFIM/Data/preprocesada/substract_local_mean_color/512x512/512x512__'
i=0
for f in aux: 
  filename = f+'.jpeg'
  ruta = ruta_img+filename
  img=cv2.imread(ruta)                # --> (row,column,channel)
  b,g,r = cv2.split(img)
  img= cv2.merge((r, g, b))
  img=img.transpose(2, 0, 1)            # --> (channel, row, column)
  img = img.reshape((1,) + img.shape)       # --> (#sample, channel, row, column)
  if i%100==0 :
    print i+1,") ",'\t', f, '\t', Y_label[i]
  X_data[i,:,:,:] = img
  i= i +1
  if i==nb_samples:
    break

print "-----------------------------"
print "Split Data..."
X_train = X_data[0:nb_train_samples]
Y_train = Y_label[0:nb_train_samples]

if same==1:
  X_test = X_train
  Y_test = Y_train
  print "SAME"
else:
  X_test = X_data[nb_train_samples:nb_samples]
  Y_test = Y_label[nb_train_samples:nb_samples]
  print "Diferente"

unique, counts = numpy.unique(Y_test, return_counts=True)
print numpy.asarray((unique, counts)).T

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print "-----------------------------"

###################################################################################

# conviertiendo en vector one hot
Y_train = np_utils.to_categorical(Y_train, nb_classes=nb_classes)
Y_test = np_utils.to_categorical(Y_test,nb_classes=nb_classes) 

print 'Image Sample format: ',X_train.shape[1:]
print 'Label Sample categorical format:' , Y_train.shape[1:]

print '------------------------------'
print 'Compile model...'
model = Sequential()

model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
#model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(16, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
  
model.add(Convolution2D(32, 3, 3, border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(96, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

model.add(Convolution2D(96, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
'''
model.add(Convolution2D(128, 3, 3,border_mode='same'))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
'''
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(96)) #W_regularizer=l2(0.00005), activity_regularizer=activity_l2(0.00005)
model.add(Activation('relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

decay = 7*1e-5
# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)  #lr=0.005 #decay=1e-6
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print(model.summary())
# Pasando X a float para hacer operaciones 

print 'Float transformation...'
#miles = nb_train_samples/1000
'''
for n in range(0,miles):
	X_train[1000*n:1000*(n+1)] = X_train[1000*n:1000*(n+1)].astype('float32')
	print (n+1)*1000 , 'OK'
'''
print '---------------------'
print X_train[1,:,120:124,120:124]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print '---------------------'
print X_train[1,:,120:124,120:124]
print '---------------------'
print 'Training...'

filepath_loss = "best-model/model-w-best-loss.hdf5"
filepath_acc = "best-model/model-w-best-acc.hdf5"
checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_loss,checkpoint_acc]

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,callbacks=callbacks_list)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zoom_range = [0.8, 1], #Zoom random crops
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size, shuffle=True),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),callbacks=callbacks_list)

opt = model.optimizer
exact_lr = opt.lr.get_value() * (1.0 / (1.0 + opt.decay.get_value() * opt.iterations.get_value()))
print exact_lr

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy model: %.2f%%" % (scores[1]*100))

train_acc = numpy.asarray(hist.history['acc'])
train_loss = numpy.asarray(hist.history['loss'])
test_acc = numpy.asarray(hist.history['val_acc'])
test_loss = numpy.asarray(hist.history['val_loss'])
epocas = []
for i in range(nb_epoch):
  epocas.append(i)

numpy.save('hist/train_acc.npy',train_acc)
numpy.save('hist/train_loss.npy',train_loss)
numpy.save('hist/test_acc.npy',test_acc)
numpy.save('hist/test_loss.npy',test_loss)

pyplot.figure(figsize=(19,12),dpi=100)
pyplot.suptitle('lr ='+str(lr)+ ' Decay='+str(decay)+' Train: '+str(nb_train_samples) + '  ' + 'Test: '+str(nb_test_samples), fontsize="x-large")

pyplot.subplot(2,2,1)
pyplot.xlabel('Epoch')
pyplot.ylabel('accuracy')
pyplot.axis([-1, nb_epoch*1.01, 0.4, 1.1])
pyplot.title('Train acc')
pyplot.plot(epocas,train_acc)
pyplot.grid()

pyplot.subplot(2,2,3)
pyplot.xlabel('Epoch')
pyplot.ylabel('loss')
pyplot.axis([-1, nb_epoch*1.01, 0, 0.8])
pyplot.title('Train loss')
pyplot.plot(epocas,train_loss)
pyplot.grid()

pyplot.subplot(2,2,2)
pyplot.xlabel('Epoch')
pyplot.ylabel('accuracy')
pyplot.axis([-1, nb_epoch*1.01, 0.4, 1.1])
pyplot.title('Test acc')
pyplot.plot(epocas,test_acc)
pyplot.grid()

pyplot.subplot(2,2,4)
pyplot.xlabel('Epoch')
pyplot.ylabel('loss')
pyplot.axis([-1, nb_epoch*1.01, 0, 0.8])
pyplot.title('Test loss')
pyplot.plot(epocas,test_loss)
pyplot.grid()

pyplot.savefig('graficas.png')

pyplot.show()


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
prediction = model.predict_classes(X_test)

pyplot.figure(figsize=(19,12),dpi=100)
pyplot.suptitle("Ground truth and prediction - 0->Ok , 1->RD ")

pyplot.subplot(1,2,2)
pyplot.axis([-0.1,1.1, -1,nb_test_samples+1 ])
pyplot.xlabel('prediction')
pyplot.ylabel('Sample')
pyplot.grid('on')
pyplot.subplot(1,2,1)
pyplot.axis([-0.1,1.1, -1,nb_test_samples+1 ])
pyplot.xlabel('ground truth')
pyplot.ylabel('Sample')
pyplot.grid('on')

for k in range(nb_test_samples):
  pyplot.subplot(1,2,2)
  pyplot.plot(numpy.asarray(prediction[k]), numpy.asarray([k]),'d')
  pyplot.subplot(1,2,1)
  pyplot.plot(Y_test[k,1], numpy.asarray([k]),'d')


pyplot.savefig('prediction.png')
#pyplot.show()


#num_img=5
#pyplot.figure()
#pyplot.imshow(toimage(X_test[num_img-1]))
#pyplot.show()
#print prediction

#aux_train = aux[0:nb_train_samples]
#aux_test = aux[nb_train_samples:nb_samples]
##aux_test = aux_train

#num_img=5
## probando con una imagen
#pyplot.figure('Test - one image')
#pyplot.imshow(toimage(X_test[num_img-1]))
#pyplot.title(aux_test[num_img-1])
#pyplot.show()
#print 'label:', Y_test[num_img-1]
#x = numpy.expand_dims(X_test[num_img-1], axis=0)
#prediction = model.predict(x)
#print 'prediction:', numpy.round(prediction)


#print "-----------------------------"
#print "Normalization: Zero Mean , standard deviation 1 , min-max 0-1"

#mean_train = X_train.mean(0)
#deviation = X_train.std(0)
#X_train=(X_train - mean_train)/deviation
#X_train = ( X_train - X_train.min(0) )/(X_train.max(0)-X_train.min(0))

#print "-----------------------------"

