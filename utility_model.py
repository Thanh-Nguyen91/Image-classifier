import torch
from torchvision import datasets
from torchvision import transforms as T
import argparse
import os

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {'train':T.Compose([T.RandomRotation(30),
                                          T.RandomResizedCrop(224),
                                          T.RandomHorizontalFlip(),
                                          T.ToTensor(),
                                          T.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                        'valid':T.Compose([T.Resize(256),
                                T.CenterCrop(224),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]),
                        'test':T.Compose([T.Resize(256),
                                T.CenterCrop(224),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])}


    image_datasets = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test':datasets.ImageFolder(test_dir, transform=data_transforms['test'])}


    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
                   'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)}

    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx


def get_input_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir',type = str, default = 'flowers/',help = 'data directory')
    parser.add_argument('--save_dir',type = str, default = str(os.getcwd()),help = 'directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg11', help = 'model architecture to use: "vgg11" or "resnet50"')
    parser.add_argument('--learnrate', type = float, default = 0.003, help = 'learning rate for trainning')
    parser.add_argument('--hidden_layers', type = int, nargs = '+', default = [500,250], help = 'hidden layers as 2 integers separated by a space, such as 500 200')
    parser.add_argument('--epochs', type = int, default = 1, help = '(int) epochs for trainning')
    parser.add_argument('--gpu', type = bool, default = False, help = '(bool) use GPU for trainning model')
    
    return parser.parse_args()
