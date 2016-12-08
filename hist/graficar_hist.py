import numpy, glob, cv2, csv, sys
from matplotlib import pyplot

train_acc = numpy.load('train_acc.npy')
train_loss = numpy.load('train_loss.npy')
test_acc = numpy.load('test_acc.npy')
test_loss = numpy.load('test_loss.npy')


lr = 0.012
decay = 7*1e-5
nb_train_samples= 15000
nb_test_samples= 2562
nb_epoch = 100

epocas = []
for i in range(nb_epoch):
  epocas.append(i)

pyplot.figure(figsize=(19,12),dpi=100)
pyplot.suptitle('lr ='+str(lr)+ ' Decay='+str(decay)+' Train: '+str(nb_train_samples) + '  ' + 'Test: '+str(nb_test_samples), fontsize="x-large")

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
pyplot.axis([-1, nb_epoch*1.01, 0, 0.8])
pyplot.title('Train loss - Test loss')
pyplot.plot(epocas,train_loss,label="train loss")
pyplot.plot(epocas,test_loss,label="test loss")
pyplot.legend(loc=2)
pyplot.grid()

pyplot.savefig('graficas.png')

pyplot.show()