import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
#from cnn_finetune import make_model
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import torch.nn as nn
import sys
sys.path.append('/home/mimiao/Desktop/vgg16/vision-master/torchvision/datasets')
import folder
#from utils import progress_bar

#fileW = open('a.log','a')
#sys.stdout = fileW

#Img_datrain = torchvision.datasets.ImageFolder('/home/mimiao/Desktop/vgg16/train', transform=transforms.ToTensor())
Img_datrain = folder.ImageFolder('/home/mimiao/Desktop/vgg16/train', transform=transforms.ToTensor())
#Img_datest = torchvision.datasets.ImageFolder('/home/mimiao/Desktop/vgg16/test', transform=transforms.ToTensor())
Img_datest = folder.ImageFolder('/home/mimiao/Desktop/vgg16/test', transform=transforms.ToTensor())
#y_train = torch.ones(len(Img_datrain))
#y_test = torch.ones(len(Img_datest))
#Img_datrain = MyDataset(txt='/home/mimiao/Desktop/vgg16/train/train.txt',transform=transforms.ToTensor())
#Img_datest = MyDataset(txt='',transform=transforms.ToTensor())
#print(Img_datrain[0])
loadertrain = DataLoader(Img_datrain, batch_size = 2, shuffle = True)
loadertest = DataLoader(Img_datest, batch_size = 2, shuffle = True)
#model = make_model('vgg16', num_class=2, pretrained=True, input_size=(540,675,3))
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.classifier = nn.Sequential(
	nn.Linear(172032, 4096),
	nn.ReLU(True),
	nn.Dropout(),
	nn.Linear(4096,4096),
	nn.ReLU(True),
	nn.Dropout(),
	nn.Linear(4096, 2))
#fc_features = vgg16.fc.in_features
vgg16.classifier._modules['6'] = nn.Linear(4096,2)
model = vgg16
#if use_cuda:
#	model.cuda()
#	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()-1))
#	cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
classes = ('polyp', 'no-polyp')
best_acc = 0
def train(epoch):
	print('\nEpoch: %d' % epoch)
	model.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets, path) in enumerate(loadertrain):
		#if use_cuda:
		#	inputs, targets = inputs.cuda(), targets.cuda()
		print(path)
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = model(inputs)
		#_,preds = torch.max(outputs.data, 1)
		#print(batch_idx)
		#print(preds)
		print(targets)
		print(outputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data,1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		#fileW.write(path)
		#fileW.write(targets)
		#fileW.write(outputs)
		#print(batch_idx, train_loss/(batch_idx+1), correct/total)
		#progress_bar(bar_idx, len(loadertrain), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
	global best_acc
	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets,path) in enumerate(loadertest):
		#if use_cuda:
		#	inputs, targets = inputs.cuda(), targets.cuda()
		print(path)
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = model(inputs)
		print(targets)
		print(outputs)
		loss = criterion(outputs,targets)
		test_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		#fileW.write(path)
		#fileW.write(targets)
		#fileW.write(outputs)
		#progress_bar(bar_idx, len(loadertest), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %(test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	acc = 100.* correct/total
	if acc > best_acc:
		print('Saving ..')
		state = {
			'net':model,
			'acc':acc,
			'epoch':epoch,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state,'./checkpoint/ckpt.t7')
		best_acc= acc
		#for batch_idx, (inputs, targets) in enumerate(loadertest):
		#	inputs, targets = Variable(inputs), Variable(targets)
		#	outputs = model(inputs)
		#	print(targets)
		#	print(outputs)
for epoch in range(0,200):
	print('Wait trainning')
	train(epoch)
	print('wait testing')
	test(epoch)
	torch.cuda.empty_cache()
