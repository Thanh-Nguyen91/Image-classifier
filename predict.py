# PROGRAMMER: Nguyen Thi Thanh
# DATE CREATED:    26/04/2020                             
# REVISED DATE: 
# PURPOSE: Classifies flower images using a pretrained deep neural network such as VGG11 and RESNET50
#          This Python script is for prediction flower name with corresponding probability
#
# Expected Call with <> indicating expected user input:
#   python predict.py --img_path <path to image> --chpt_path <path to model checkpoint>
#             --top_k <top K most likely classes> --cat_name <file for label mapping>
# Example call:
#   python predict.py --img_path rose.jpeg --chpt_path checkpoint.pth  --top_k 5 --cat_name cat_to_name.json --gpu True
##

from fc_model import rebuild_model
import utility_predict as U
import json
import torch


## get options from input args
in_arg = U.get_input_arg()
img_path = in_arg.img_path
chpt_path = in_arg.chpt_path
top_k = in_arg.top_k
cat_name = in_arg.cat_name
gpu = in_arg.gpu

print('\n*** Input arguments ***\n\npath to image: {}\npath to checkpoint: {}\ntop K = {}\ncategory to name by {}\nGPU use {}\n'.format(img_path,chpt_path,top_k,cat_name,gpu))

## load model from a checkpoint
print('\n*** Loading model ***\n')
model = rebuild_model(chpt_path)

if gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    

## predict and print out result
print('\n*** Predicting class ***\n\n')      
with open(cat_name, 'r') as f:
    cat_to_name = json.load(f)
    
top_p,top_class,top_flowers=U.predict(img_path, model, top_k,cat_to_name,device)

print('{:<3}\t{:<20}\t{:<5}\n'.format('ID','Name','Proba'))
form = '{:<3}\t{:<20}\t{:<5.2f}'
for prob,idx,name in zip(top_p,top_class,top_flowers):
    print(form.format(idx,name,prob))
