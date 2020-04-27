#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision import transforms as T
from PIL import Image
import argparse


def get_input_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--img_path',type = str, default = None,help = '(str) path to the image')
    parser.add_argument('--chpt_path', type = str, default = None, help = '(str) path to model checkpoint')
    parser.add_argument('--top_k', type = int, default = 5, help = '(int) top K of most likely classes')
    parser.add_argument('--cat_name', type = str, default = None, help = '(str) file for mapping the category to real name')
    parser.add_argument('--gpu', type = bool, default = False, help = '(bool) use GPU for predicting class')
    
    return parser.parse_args()


def process_image(image_path):
    """ Process image before putting in model, return a Pytorch Tensor
    """
    img = Image.open(image_path)
    transform = T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img = transform(img)
    
    return img


def predict(image_path, model, topk,cat_to_name,device):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    
    image = process_image(image_path)
    
    model,image = model.to(device),image.to(device)
    model.eval()
    logps = model(image.unsqueeze(0))
    ps = torch.exp(logps)
    top_p,top_class = ps.topk(topk,dim=1)
    
       
    # convert to numpy array
    np_top_p = top_p.cpu()[0].detach().numpy()
    np_top_class = top_class.cpu()[0].numpy()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class_id = [idx_to_class[clss] for clss in np_top_class]
    top_flowers = [cat_to_name[str(idx)] for idx in top_class_id]    
   
    return np_top_p,np_top_class,top_flowers
