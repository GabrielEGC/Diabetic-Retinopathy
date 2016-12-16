import numpy, glob, cv2, csv, sys
from matplotlib import pyplot

train_acc = numpy.load('train_acc.npy')
train_loss = numpy.load('train_loss.npy')
test_acc = numpy.load('test_acc.npy')
test_loss = numpy.load('test_loss.npy')

max_index = numpy.argmax(test_acc)
test_max_acc = test_acc[max_index]
loss = test_loss[max_index] 

min_index = numpy.argmin(test_loss)
test_min_loss = test_loss[min_index]
acc = test_acc[min_index]

red = 'VGG noFC'
lr = 0.0001
decay = 0
nb_train_samples= 4000*2 #15000
nb_test_samples= 692*2 #2562
nb_epoch = 100

epocas = []
for i in range(1,nb_epoch+1):
	  epocas.append(i)

pyplot.figure(figsize=(19,12),dpi=100)
pyplot.suptitle(red + ' - ' +'lr ='+str(lr)+ ' - ' +'[Train Validation]= '+'['+str(nb_train_samples) +' ' +str(nb_test_samples)+']' + ' - ' + 
				'Max val ACC: %.2f' % (test_max_acc*100) + '%' + ' - ' + ' Epoch: ' + str(max_index+1), fontsize="x-large")

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
pyplot.legend(loc=3)
pyplot.grid()

pyplot.savefig('graficas.png')


print ' '
print '############ Max val Accuracy ############'
print 'Max val ACC: ', test_max_acc , ' Epoch: ', max_index+1, ' its loss: ', loss,  ' its train acc: ' ,train_acc[max_index]  , ' its train loss: ', train_loss[max_index] 
print ' '
print '############ Min val Loss ############'
print 'Min val LOSS: ', test_min_loss, ' Epoch: ', min_index +1, ' its acc: ', acc, ' its train acc: ', train_acc[min_index]  , ' its train loss: ', train_loss[min_index]
print ' '

pyplot.show()