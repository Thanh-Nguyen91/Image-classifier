# PROGRAMMER: Nguyen Thi Thanh
# DATE CREATED:    26/04/2020                             
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pretrained deep neural network such as VGG11 and RESNET50
#          This Python script is used to build and train a new classifier of the pretrained model (VGG11 as defaut)
#
# Expected Call with <> indicating expected user input:
#   python train.py --data_dir <directory with images> --arch <model> --hidden_layers <list of hidden layers size>
#             --learnrate <learning rate> --epochs <trainning epochs> --save_dir <directory to save the checkpoint>
# Example call:
#   python train.py --data_dir flowers/ --arch vgg11  --hidden_layers 500 250 --learnrate 0.003 --epochs 2 --save_dir checkpoints/ --gpu True
##


import torch
from torch import nn,optim
from torchvision import models
import fc_model
from utility_model import load_data, get_input_arg


## get options from input args
in_arg = get_input_arg()
data_dir = in_arg.data_dir
save_dir = in_arg.save_dir
arch = in_arg.arch
learnrate = in_arg.learnrate
hidden_layers= in_arg.hidden_layers
epochs = in_arg.epochs
gpu = in_arg.gpu
  

print('\n*** Input arguments ***\n\ndata directory: {}\nmodel architecture: {} -- hidden layers: {} -- learning rate: {}  -- epochs: {}\nsave checkpoint to {}\nGPU use: {}'.format(data_dir,arch,hidden_layers,learnrate,epochs,save_dir,gpu))

## load data
print('\n*** Loading data ***\n')
dataloaders,class_to_idx = load_data(data_dir)

output_size = 102

dropout=0.2

## buil model
print('\n*** Building model ***\n\n')
if arch == 'vgg11':
    model = models.vgg11(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
        
    input_size = 25088
    model.classifier = fc_model.classifier(input_size,output_size,hidden_layers,dropout)
elif arch=='resnet50':
    model = models.resnet50(pretrained=True)
    for para in model.parameters():
        para.requires_grad = False
    input_size = 2048
    model.fc = fc_model.classifier(input_size,output_size,hidden_layers,dropout)

print('Model built\n')
#print(model,'\n')


## train model
print('\n*** Trainning model ***\n\n')
if gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
criterion = nn.NLLLoss()
if arch=='vgg11':
    optimizer = optim.Adam(model.classifier.parameters(),lr=learnrate)
elif arch =='resnet50':
    optimizer = optim.Adam(model.fc.parameters(),lr=learnrate)
    
model,train_losses,valid_losses = fc_model.train(model,criterion,optimizer,dataloaders['train'],dataloaders['valid'],epochs,device)


## save the checkpoint
print('\n*** saving model checkpoint ***')
model.class_to_idx = class_to_idx

checkpoint = {'arch':arch,
              'input_size': input_size,
              'output_size': output_size,
              'hidden_layers': hidden_layers,
              'dropout':dropout,
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx,
              'epochs':epochs,
              'optimizer_state':optimizer.state_dict()
               }

torch.save(checkpoint, save_dir+'/checkpoint.pth')
