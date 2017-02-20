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


def data_50_50(aux, Y_label,nb_train_samples,nb_val_samples, nb_test_samples):
	
	num_zero_train= nb_train_samples/2
	num_one_train = nb_train_samples/2

	num_zero_val= nb_val_samples/2
	num_one_val = nb_val_samples/2

	num_zero_test = nb_test_samples/2
	num_one_test = nb_test_samples/2

	var = numpy.asarray([aux,Y_label])
	var = var.T

	data_0 = var[Y_label==0]
	data_1 = var[Y_label>0]

	data_t_0 = data_0[0:num_zero_train]
	data_t_1 = data_1[0:num_one_train]

	data_train = numpy.concatenate((data_t_0,data_t_1),axis=0)
	numpy.random.shuffle(data_train)

	data_val_0 = data_0[num_zero_train:(num_zero_train + num_zero_val)]
	data_val_1 = data_1[num_one_train:(num_one_train + num_one_val)]

	data_val = numpy.concatenate((data_val_0,data_val_1),axis=0)
	numpy.random.shuffle(data_val)

	data_test_0 = data_0[(num_zero_train + num_zero_val):(num_zero_train + num_zero_val + num_zero_test)]
	data_test_1 =data_1[(num_one_train + num_one_val):(num_one_train + num_one_val + num_one_test)]

	data_test = numpy.concatenate((data_test_0,data_test_1),axis=0)
	numpy.random.shuffle(data_test)
	#print 'train samples' , len(data_t_0)+len(data_t_1) #+ data_t_1.shape(0)
	#print 'val samples' , len(data_val_0)+len(data_val_1)#+ data_val_1.shape(0)
	#print 'test samples' , len(data_test_0)+len(data_test_1) #+ data_test_1.shape(0)

	data = numpy.concatenate((data_train,data_val,data_test),axis=0)

	print 'TOTAL', len(data)
	aux = data[:,0] #nombre de imagenes
	Y_label = numpy.uint8(data[:,1])
	return (aux, Y_label)
	'''
	train samples 3692*2  = 7384
	val samples  500*2 = 1000
	test samples  500*2 = 1000
	'''

	# GET DATA INFO

	'''
	left (de 1 a 2.74)
	0: 12870  1: 4692

	Right (de 1 a 2.79)
	0: 12938  1:4624
	'''


def split_data(X_data, Y_label,nb_train_samples,  nb_val_samples,nb_test_samples, nb_classes, same):
	X_train = X_data[0:nb_train_samples]
	Y_train = Y_label[0:nb_train_samples]

	if same==1:
  		X_val = X_train
  		Y_val = Y_train
  		print "Same X_train X_val"
	else:
  		X_val = X_data[nb_train_samples:(nb_train_samples+nb_val_samples)]
  		Y_val = Y_label[nb_train_samples:(nb_train_samples+nb_val_samples)]
  		print "Different X_train X_val"

  	X_test = X_data[(nb_train_samples+nb_val_samples):(nb_train_samples+nb_val_samples+nb_test_samples)]
  	Y_test = Y_label[(nb_train_samples+nb_val_samples):(nb_train_samples+nb_val_samples+nb_test_samples)]

	print 'Data Train'
	unique, counts = numpy.unique(Y_train, return_counts=True)
	print numpy.asarray((unique, counts)).T

	print 'Data Val'
	unique, counts = numpy.unique(Y_val, return_counts=True)
	print numpy.asarray((unique, counts)).T

	print 'Data Test'
	unique, counts = numpy.unique(Y_test, return_counts=True)
	print numpy.asarray((unique, counts)).T

	#print('X_train shape:', X_train.shape)
	#print('X_val shape:' , X_val.shape)
	#print('X_test shape:', X_test)
	print(X_train.shape[0], ' train samples')
	print(X_val.shape[0], ' val samples')
	print(X_test.shape[0], ' test samples')

	return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

	

def get_data(img_rows, img_cols,side,  same, data_per_classes,nb_classes):
	etiquetas=17562
	img_channels = 3
	# DATA SPLIT
	if data_per_classes == 0:
		print "______DIST ORIGINAL______"
		nb_train_samples= 12294#14162 
		nb_val_samples = 2634#1700 
		nb_test_samples= 2634#1700 
		nb_samples=nb_train_samples + nb_val_samples + nb_test_samples
	elif data_per_classes ==1:
		print "______DIST 50/50______"
		nb_train_samples= 3692*2 #4000*2
		nb_val_samples = 500*2 #692*2
		nb_test_samples= 500*2 
		nb_samples=nb_train_samples + nb_val_samples + nb_test_samples


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

	'''
	#Shuffle DATA
	var = numpy.asarray([aux,Y_label])
	var = var.T
	numpy.random.shuffle(var)
	aux = var[:,0] #nombre de imagenes
	Y_label = numpy.uint8(var[:,1])
	'''		
	if data_per_classes==1:
		(aux, Y_label) = data_50_50(aux, Y_label, nb_train_samples, nb_val_samples,nb_test_samples)
	
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
  		if i%500==0 :
			print i+1,") ",'\t', f, '\t', Y_label[i]
  		X_data[i,:,:,:] = img
  		i= i +1
  		if i==nb_samples:
			break
	(X_train, Y_train,X_val, Y_val, X_test, Y_test) = split_data(X_data, Y_label,nb_train_samples, nb_val_samples,nb_test_samples , nb_classes, same)

    # conviertiendo en vector one hot
	Y_train = np_utils.to_categorical(Y_train, nb_classes=nb_classes)
	Y_val = np_utils.to_categorical(Y_val, nb_classes=nb_classes)
	Y_test = np_utils.to_categorical(Y_test,nb_classes=nb_classes) 

	print 'Image Sample format: ',X_train.shape[1:]
	print 'Label Sample categorical format:' , Y_train.shape[1:]

	print 'Float transformation...'
	print '---------------------'
	print X_val[1,:,120:124,120:124]
	X_train = X_train.astype('float32')
	X_val = X_val.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_val /= 255
	X_test /= 255
	print '---------------------'
	print X_val[1,:,120:124,120:124]
	print '---------------------'
	return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def compile_model(model, lr, decay, momentum, nesterov):
	# let's train the model using SGD + momentum (how original).
	sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)  #lr=0.005 #decay=1e-6
	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])
	print(model.summary())
	
	return model

def train_model(model, X_train, Y_train, X_val,Y_val,X_test, Y_test,batch_size, nb_epoch,data_augmentation):
	filepath_loss = "best-model/model-w-best-loss.hdf5"
	filepath_acc = "best-model/model-w-best-acc.hdf5"
	checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')
	checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
	callbacks_list = [checkpoint_loss,checkpoint_acc]

	#Activate with the original distribution of data (data_per_classes =0)
	class_weight={0:1. , 1:2.74}

	if not data_augmentation:
		print('Not using data augmentation.')
		hist = model.fit(X_train, Y_train, 
						batch_size=batch_size, nb_epoch=nb_epoch, 
						validation_data=(X_val, Y_val), 
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
	                        validation_data=(X_val, Y_val),callbacks=callbacks_list, class_weight=class_weight)

	#scores = model.evaluate(X_test, Y_test, verbose=0)
	#print("Accuracy model: %.2f%%" % (scores[1]*100))
	return hist

def plot_hist(hist, nb_epoch, name,Y_train, Y_val, Y_test, DA):
	train_samples = len(Y_train)
	val_samples = len(Y_val)
	test_samples = len(Y_test)

	train_acc = numpy.asarray(hist.history['acc'])
	train_loss = numpy.asarray(hist.history['loss'])
	val_acc = numpy.asarray(hist.history['val_acc'])
	val_loss = numpy.asarray(hist.history['val_loss'])
	epocas = []
	for i in range(1,nb_epoch+1):
	  epocas.append(i)

	numpy.save('hist/train_acc.npy',train_acc)
	numpy.save('hist/train_loss.npy',train_loss)
	numpy.save('hist/val_acc.npy',val_acc)
	numpy.save('hist/val_loss.npy',val_loss)

	max_index = numpy.argmax(val_acc)
	val_max_acc = val_acc[max_index]
	loss = val_loss[max_index] 

	min_index = numpy.argmin(val_loss)
	val_min_loss = val_loss[min_index]
	acc = val_acc[min_index]

	pyplot.figure(figsize=(19,12),dpi=100)
	pyplot.suptitle(name + ' - ' +'lr = '+str(lr)+' - ' +'[Train Validation Test]= '+'['+str(train_samples) +' ' +str(val_samples)+' ' +str(test_samples)+']' 
					+ ' - ' + 'Max val acc: %.2f' % (val_max_acc*100) + '%' + ' - ' + ' Epoch: ' + str(max_index+1)
					+ '- ' + 'DA = ' + str(DA))#fontsize="x-large"

	pyplot.subplot(2,1,1)
	pyplot.xlabel('Epoch')
	pyplot.ylabel('accuracy')
	pyplot.axis([-1, nb_epoch*1.01, 0.4, 1.1])
	pyplot.title('Train acc - Val acc')
	pyplot.plot(epocas,train_acc, label="train acc")
	pyplot.plot(epocas,val_acc, label="val acc")
	pyplot.legend(loc=2)
	pyplot.grid()

	pyplot.subplot(2,1,2)
	pyplot.xlabel('Epoch')
	pyplot.ylabel('loss')
	pyplot.axis([-1, nb_epoch*1.01, 0, 1])
	pyplot.title('Train loss - Val loss')
	pyplot.plot(epocas,train_loss,label="train loss")
	pyplot.plot(epocas,val_loss,label="val loss")
	pyplot.legend(loc=3)
	pyplot.grid()

	pyplot.savefig('graficas.png')

	print ' '
	print '############ Max Val Accuracy ############'
	print 'Max Val ACC: ', val_max_acc , ' Epoch: ', max_index+1, ' its loss: ', loss,  ' its train acc: ' ,train_acc[max_index]  , ' its train loss: ', train_loss[max_index] 
	print ' '
	print '############ Min Val Loss ############'
	print 'Min Val LOSS: ', val_min_loss, ' Epoch: ', min_index +1, ' its acc: ', acc, ' its train acc: ', train_acc[min_index]  , ' its train loss: ', train_loss[min_index]
	print ' '

	pyplot.show()

def extra_values(X_test, Y_test):
	TP = 0.0
	TN = 0.0
	FP = 0.0
	FN = 0.0
	n = len(X_test)
	print n
	for i in range(0,n):
		label = Y_test[i][0]
		x = numpy.expand_dims(X_test[i], axis=0)
		prediction = model.predict(x)
		prediction = numpy.round(prediction)
		prediction = prediction[0][0]
		if label==1.0 and label==prediction:
			#TRUE POSITIVE
			TP = TP+1.0
		elif label==1.0 and label!=prediction:
			#FALSE NEGATIVE
			FN = FN + 1.0
		elif label==0.0 and label==prediction:
			#TRUE NEGATIVE
			TN = TN + 1.0
		elif label == 0.0 and label!=prediction:
			#FALSE POSITIVE
			FP = FP + 1.0

	print 'TRUE POSITIVE: ', TP
	print 'FALSE NEGATIVE: ', FN
	print 'TRUE NEGATIVE: ', TN
	print 'FALSE POSITIVE: ', FP
	print '----------------------'
	print 'Sensitivity: ', TP / (TP + FN) * 100
	print 'Specificity: ', TN / (FP + TN) * 100

###________MAIN____________###
if __name__ == "__main__":
	
	# GET DATA INFO

	'''
	left (de 1 a 2.74)
	0: 12870  1: 4692

	Right (de 1 a 2.79)
	0: 12938  1:4624
	'''

	img_rows, img_cols = 256, 256 # input image dimensions
	same=0 #Flag Xtrain = Xval   otherwise 0 =->Xtrain and val different
	data_per_classes =0 #flag activar distribcion 50/50
	side = 'left'

	# CREATE MODEL INFO
	nb_classes = 2
	
	# TRAIN MODEL INFO	
	lr = 0.0001
	decay = 0.00005
	momentum = 0.9 
	nesterov = True
	batch_size = 64
	nb_epoch = 80
	data_augmentation = True 
	entrenar = 0

	print "####################### GET DATA ###############################"
	(X_train, Y_train, X_val, Y_val, X_test, Y_test) = get_data(img_rows=img_rows, img_cols=img_cols, side=side, same=same, data_per_classes=data_per_classes, nb_classes=nb_classes)
	input_shape = X_train.shape[1:]
	print "####################### CREATE MODEL ###########################"
	model, name = vgg_net_noFC(input_shape = input_shape,nb_classes= nb_classes)
	
	if entrenar == 1:
		print "Compile model"
		model = compile_model(model = model, lr = lr, decay = decay, momentum = momentum, nesterov = nesterov)
		print "####################### TRAIN MODEL ###########################"
		hist = train_model(model = model, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,X_test=X_test, Y_test=Y_test, batch_size=batch_size, nb_epoch=nb_epoch, data_augmentation=data_augmentation)
		print "####################### GRAPHICS ###############################"
		#Graphics#
		plot_hist(hist=hist,nb_epoch=nb_epoch, name=name, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, DA = data_augmentation)
	elif entrenar == 0:
		print "###################### TRY TEST ###############################"
		print "Import weights"
		model.load_weights('best-model/model-w-best-acc.hdf5')
		print "Compile model"
		model = compile_model(model = model, lr = lr, decay = decay, momentum = momentum, nesterov = nesterov)
		scores = model.evaluate(X_test, Y_test, verbose=0)
		print("Accuracy model: %.2f%%" % (scores[1]*100))
		extra_values(X_test=X_test, Y_test=Y_test)
