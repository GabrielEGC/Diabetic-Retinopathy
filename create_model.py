from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.applications.vgg16 import VGG16

def model_1(input_shape, nb_classes):
	model = Sequential()

	model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=input_shape))
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

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(96)) #W_regularizer=l2(0.00005), activity_regularizer=activity_l2(0.00005)
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	print "MODEL 1"
	return model, "MODEL 1"

def model_2(input_shape, nb_classes):
	model = Sequential()

	model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=input_shape))
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

	model.add(Convolution2D(128, 3, 3,border_mode='same'))
	#model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(96)) #W_regularizer=l2(0.00005), activity_regularizer=activity_l2(0.00005)
	model.add(Activation('relu'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	print "MODEL 2"
	return model, "MODEL 2"

def vgg_net(input_shape, nb_classes):
	model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
	x = Flatten()(model.output)
	x = Dense(4096, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(4096, activation='relu')(x)
	x = Dropout(0.5)(x)
	top_model = Dense(nb_classes, activation='softmax')(x)
	model = Model(input=model.input, output=top_model)
	print "VGG16"
	return model, "VGG16"