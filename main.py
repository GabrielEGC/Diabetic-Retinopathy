import numpy, csv, cv2, glob, sys
import os    
os.environ['THEANO_FLAGS'] = "device=gpu0,floatX=float32"

# fix random seed for reproducibility
numpy.random.seed(1337)

from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from create_model import *
K.set_image_dim_ordering('th')


def data_50_50(aux, Y_label, num_zero_class,num_one_class,nb_train_samples,nb_test_samples):
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
		#numpy.random.shuffle(data)
		aux = data[:,0] #nombre de imagenes
		Y_label = numpy.uint8(data[:,1])
	else:
		sys.exit("Error - (num_zero_class + num_one_class) diferente de nb_train_samples -- Line 35 - 47")
	return (aux, Y_label)

def split_data(X_data, Y_label,nb_train_samples, nb_samples, nb_classes, same):
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

	print 'Data Train'
	unique, counts = numpy.unique(Y_train, return_counts=True)
	print numpy.asarray((unique, counts)).T
	print 'Data Test'
	unique, counts = numpy.unique(Y_test, return_counts=True)
	print numpy.asarray((unique, counts)).T

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	return (X_train, Y_train, X_test, Y_test)

	

def get_data(nb_train_samples,nb_test_samples, etiquetas, side, img_channels, img_rows, img_cols,  same, data_per_classes,nb_classes):
	nb_samples = nb_train_samples + nb_test_samples

	X_data = numpy.zeros((nb_samples, img_channels, img_rows, img_cols), dtype="uint8")
	Y_label = numpy.zeros((etiquetas,), dtype="uint8")
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
		num_zero_class= nb_train_samples/2
		num_one_class = nb_train_samples/2
		(aux, Y_label) = data_50_50(aux, Y_label, num_zero_class, num_one_class, nb_train_samples, nb_test_samples)

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
	(X_train, Y_train, X_test, Y_test) = split_data(X_data, Y_label,nb_train_samples, nb_samples, nb_classes, same)

    # conviertiendo en vector one hot
	Y_train = np_utils.to_categorical(Y_train, nb_classes=nb_classes)
	Y_test = np_utils.to_categorical(Y_test,nb_classes=nb_classes) 

	print 'Image Sample format: ',X_train.shape[1:]
	print 'Label Sample categorical format:' , Y_train.shape[1:]

	print 'Float transformation...'
	print '---------------------'
	print X_train[1,:,120:124,120:124]
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	print '---------------------'
	print X_train[1,:,120:124,120:124]
	print '---------------------'
	return (X_train, Y_train, X_test, Y_test)

def compile_model(model, lr, decay, momentum, nesterov):
	# let's train the model using SGD + momentum (how original).
	sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)  #lr=0.005 #decay=1e-6
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	print(model.summary())
	
	return model

def train_model(model, X_train, Y_train, X_test, Y_test,batch_size, nb_epoch,data_augmentation):
	filepath_loss = "best-model/model-w-best-loss.hdf5"
	filepath_acc = "best-model/model-w-best-acc.hdf5"
	checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')
	checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
	callbacks_list = [checkpoint_loss,checkpoint_acc]

	if not data_augmentation:
		print('Not using data augmentation.')
		hist = model.fit(X_train, Y_train, 
						batch_size=batch_size, nb_epoch=nb_epoch, 
						validation_data=(X_test, Y_test), 
						shuffle=True,callbacks=callbacks_list)
	else:
	    print('Using real-time data augmentation.')

	    # this will do preprocessing and realtime data augmentation
	    datagen = ImageDataGenerator(
	        rotation_range=0,
	        zoom_range=[0.8, 1],
	        fill_mode='nearest',
	        horizontal_flip=False,
	        vertical_flip=True,
	        dim_ordering=K.image_dim_ordering())  # randomly flip images

	    # compute quantities required for featurewise normalization
	    # (std, mean, and principal components if ZCA whitening is applied)
	    datagen.fit(X_train)

	    # fit the model on the batches generated by datagen.flow()
	    hist = model.fit_generator(datagen.flow(X_train, Y_train,
	                        batch_size=batch_size,shuffle=True), #save_to_dir='da'
	                        samples_per_epoch=X_train.shape[0],
	                        nb_epoch=nb_epoch,
	                        validation_data=(X_test, Y_test),callbacks=callbacks_list)

	scores = model.evaluate(X_test, Y_test, verbose=0)
	print("Accuracy model: %.2f%%" % (scores[1]*100))
	return hist

def plot_hist(hist, nb_epoch, name):
	train_acc = numpy.asarray(hist.history['acc'])
	train_loss = numpy.asarray(hist.history['loss'])
	test_acc = numpy.asarray(hist.history['val_acc'])
	test_loss = numpy.asarray(hist.history['val_loss'])
	epocas = []
	for i in range(1,nb_epoch+1):
	  epocas.append(i)

	numpy.save('hist/train_acc.npy',train_acc)
	numpy.save('hist/train_loss.npy',train_loss)
	numpy.save('hist/test_acc.npy',test_acc)
	numpy.save('hist/test_loss.npy',test_loss)

	pyplot.figure(figsize=(19,12),dpi=100)
	pyplot.suptitle(name + ' - ' +'lr ='+str(lr)+' - ' +'[Train Validation]= '+'['+str(nb_train_samples) +' ' +str(nb_test_samples)+']', fontsize="x-large")

	pyplot.subplot(2,1,1)
	pyplot.xlabel('Epoch')
	pyplot.ylabel('accuracy')
	pyplot.axis([-1, nb_epoch*1.01, 0.4, 1.1])
	pyplot.title('Train acc - Test acc')
	pyplot.plot(epocas,train_acc, label="train acc")
	pyplot.plot(epocas,test_acc, label="test acc")
	pyplot.legend(loc=2)
	pyplot.grid()

	pyplot.subplot(2,1,2)
	pyplot.xlabel('Epoch')
	pyplot.ylabel('loss')
	pyplot.axis([-1, nb_epoch*1.01, 0, 1])
	pyplot.title('Train loss - Test loss')
	pyplot.plot(epocas,train_loss,label="train loss")
	pyplot.plot(epocas,test_loss,label="test loss")
	pyplot.legend(loc=3)
	pyplot.grid()

	pyplot.savefig('graficas.png')

	max_index = numpy.argmax(test_acc)
	test_max_acc = test_acc[max_index]
	loss = test_loss[max_index] 

	min_index = numpy.argmin(test_loss)
	test_min_loss = test_loss[min_index]
	acc = test_acc[min_index]
	print ' '
	print '############ Max Val Accuracy ############'
	print 'Max Val ACC: ', test_max_acc , ' Epoch: ', max_index+1, ' its loss: ', loss,  ' its train acc: ' ,train_acc[max_index]  , ' its train loss: ', train_loss[max_index] 
	print ' '
	print '############ Min Val Loss ############'
	print 'Min Val LOSS: ', test_min_loss, ' Epoch: ', min_index +1, ' its acc: ', acc, ' its train acc: ', train_acc[min_index]  , ' its train loss: ', train_loss[min_index]
	print ' '

	pyplot.show()


###________MAIN____________###
if __name__ == "__main__":
	
	# GET DATA INFO

	'''
	left (de 1 a 2.74)
	0: 12870  1: 4692

	Right (de 1 a 2.79)
	0: 12938  1:4624
	'''

	nb_train_samples= 4000*2 #15000
	nb_test_samples= 692*2 #2562
	side = "left"
	etiquetas = 17562 #numleft = 17562 #numrigth = 17562 <------> labels
	nb_samples=nb_train_samples + nb_test_samples
	img_rows, img_cols = 256, 256 # input image dimensions
	img_channels = 3
	same=0 #Flag Xtrain = Xtest   otherwise 0 =->Xtrain and test different
	data_per_classes =1 #flag activar distribcion 50/50

	# CREATE MODEL INFO
	nb_classes = 2
	
	# TRAIN MODEL INFO	
	lr = 0.0001
	decay = 0
	momentum = 0.9
	nesterov = True
	batch_size = 64
	nb_epoch = 100
	data_augmentation = True 

	print "####################### GET DATA ###############################"
	(X_train, Y_train, X_test, Y_test) = get_data(nb_train_samples=nb_train_samples, nb_test_samples=nb_test_samples, 
												   etiquetas = etiquetas, side=side, img_channels=img_channels,img_rows=img_rows,
												   img_cols=img_cols,same=same,data_per_classes=data_per_classes, nb_classes=nb_classes)
	input_shape = X_train.shape[1:]
	print "####################### CREATE MODEL ###########################"
	model, name = vgg_net_noFC(input_shape = input_shape,nb_classes= nb_classes)
	model = compile_model(model = model, lr = lr, decay = decay, momentum = momentum, nesterov = nesterov)
	print "####################### TRAIN MODEL ###########################"
	hist = train_model(model = model, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, batch_size=batch_size, nb_epoch=nb_epoch, data_augmentation=data_augmentation)
	print "####################### GRAPHICS ###############################"
	#Graphics#
	plot_hist(hist=hist,nb_epoch=nb_epoch, name=name)
	