#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict


def classifier(input_size,output_size,hidden_layers,dropout):
    
        
    classifier = nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(input_size,hidden_layers[0])),
                                 ('relu1',nn.ReLU()),
                                 ('dropout1',nn.Dropout(dropout)),
                                 ('fc2',nn.Linear(hidden_layers[0],hidden_layers[1])),
                                 ('relu2',nn.ReLU()),
                                 ('dropout2',nn.Dropout(dropout)),
                                 ('fc3',nn.Linear(hidden_layers[1],output_size)),
                                 ('logsm',nn.LogSoftmax(dim=1))]))

        
    return classifier


def validation(model,criterion,validloader,device='cpu'):
    
    valid_loss = 0
    accuracy = 0
    
    model.to(device) # move model to device

    model.eval() # set model to evaluation mode (without dropout)
    
    # turn off gradient
    with torch.no_grad():
        for images,labels in validloader:
            images,labels = images.to(device),labels.to(device)
            logps = model(images)
            loss = criterion(logps,labels)
            valid_loss += loss.item()

            # compute accuracy
            ps = torch.exp(logps)
            top_p,top_class = ps.topk(1,dim=1)
            equals = top_class==labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return valid_loss/len(validloader),accuracy/len(validloader)



def train(model,criterion,optimizer,trainloader,validloader,epochs=10,device='cpu'):
    
    model.to(device) # move model to device
    
    print_every = 5 # print result after every few steps
    train_losses, valid_losses = [], [] # loss after every epoch
    
    for e in range(epochs):
        ## TRAIN
        train_loss = 0
        step = 0
        
        for images,labels in trainloader:
            step +=1
            images,labels = images.to(device),labels.to(device)
            
            optimizer.zero_grad() # clear gradient in optimizer

            logps = model(images)
            loss = criterion(logps,labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            ## VALIDATION
            if step % print_every ==0 or step==len(trainloader):
                valid_loss,accuracy = validation(model,criterion,validloader,device)

                print('Epoch {:2}/{:2}'.format(e+1,epochs),
                      'Step {:3}/{:3}'.format(step,len(trainloader)),
                      'Trainning loss: {:<5.3f}'.format(train_loss/step),
                      'Validation loss: {:<5.3f}'.format(valid_loss),
                      'Accuracy: {:<5.2f}%'.format(100*accuracy))
                      
        # save train and validation loss
        train_losses.append(train_loss/len(trainloader))
        valid_losses.append(valid_loss)

    return model,train_losses,valid_losses


def rebuild_model(filepath):
    chpt = torch.load(filepath, map_location=lambda storage, loc: storage)

    classifier_ = classifier(chpt['input_size'],chpt['output_size'],chpt['hidden_layers'],chpt['dropout'])
    
    if chpt['arch']=='vgg11':
        model = models.vgg11(pretrained=True)
        for para in model.parameters():
            para.requires_grad = False
        model.classifier = classifier_
        
    elif chpt['arch']=='resnet50':
        model = models.resnet50(pretrained=True)
        for para in model.parameters():
            para.requires_grad = False
        model.fc = classifier_   
        
    model.class_to_idx = chpt['class_to_idx']
    model.load_state_dict(chpt['state_dict'])
        
    return model
    
