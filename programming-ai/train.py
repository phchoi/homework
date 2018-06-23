#!/usr/bin/env python3.5


import os
import torch
import time
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms, models


class Network(object):
    def define_network(self):
        if self.args['arch'] == 'vgg13':
            model = models.vgg13(pretrained=True)
            input_size = 25088
        if self.args['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_size = 25088
        if self.args['arch'] == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_size = 9216
        return model, input_size
    
    def define_classifier(self):
        input_size = self.input_size
        output_size = self.output_size
        hidden_layers = self.args['hidden_units']
        layers = OrderedDict()
        layers['fc1'] = nn.Linear(input_size, hidden_layers[0])
        layers['relu1'] = nn.ReLU()
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        fc = 2
        count = 0 
        for h1, h2 in layer_sizes:
            if count < len(hidden_layers):
                fckey = 'fc'+str(fc)
                relukey = 'relu'+str(fc)
                layers[fckey] = nn.Linear(h1,h2)
                layers[relukey] = nn.ReLU()
                count = count + 1
                fc = fc + 1
        key = 'fc'+str(fc)
        layers[key] = nn.Linear(hidden_layers[-1], output_size)
        layers['output'] = nn.LogSoftmax(dim=1)
        classifier = nn.Sequential(layers)
            
        return classifier

    def activate_cuda_if_available(self):
        if self.args['gpu'] and torch.cuda.is_available():
            # Move model parameters to the GPU
            self.model.cuda()
        else:
            self.model.cpu()
        return True

    def save_checkpoint(self):
        self.model.class_to_idx = self.data_loader.train_datasets.class_to_idx

        checkpoint = {'model': self.model,
                      'input_size': self.input_size,
                      'output_size': self.output_size,
                      'hidden_layers': self.args['hidden_units'],
                      'state_dict': self.model.state_dict()}

        checkpoint_file = self.args['save_dir'] + self.args['checkpoint_name']
        torch.save(checkpoint, checkpoint_file)
        #torch.save(checkpoint, self.args['save_checkpoint'])
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.args['load_checkpoint'])
        self.model = checkpoint['model']
        classifier = create_custom_classifier(checkpoint['input_size'],
                                              checkpoint['output_size'],
                                              checkpoint['hidden_layers'])
        self.model.classifier = classifier
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def train(self):
        epochs = self.args['epoch']
        test_loaders = self.data_loader.test_loaders
        train_loaders = self.data_loader.train_loaders
        print_every = 10
        steps = 0 
        running_loss = 0
        criterion = nn.NLLLoss()
        if self.gpu_enabled: criterion.cuda()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

        start_time = time.time()
        for e in range(epochs):
            self.model.train()
            for images, labels in iter(train_loaders):
                steps += 1

                inputs = Variable(images)
                targets = Variable(labels)
                
                if self.gpu_enabled: inputs, targets = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                
                output = self.model.forward(inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    self.model.eval()
                    
                    accuracy = 0
                    test_loss = 0
                    
                    for images, labels in iter(test_loaders):

                        with torch.no_grad():
                            images, labels = Variable(images), Variable(labels)
        
                        if self.gpu_enabled: images, labels = images.cuda(), labels.cuda()
        
                        output = self.model.forward(images)
                        test_loss += criterion(output, labels).item()
        
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
        
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
                    elapsed_time = time.time() - start_time
        
                    print("Epoch: {}/{},".format(e+1, epochs),
                          "Steps: {},".format(steps),
                          "Elapsed: {:.3f} secs,".format(elapsed_time),
                          "Training Loss: {:.3f},".format(running_loss/print_every),
                          "Test Loss: {:.3f},".format(test_loss/len(test_loaders)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(test_loaders)))
        
                    running_loss = 0
                    self.model.train()
    
    def get_output_size(self):
        data_dir = self.args['datadir']
        train_dir = data_dir + '/' + self.args['train_dir']
        valid_dir = data_dir + '/' + self.args['valid_dir']
        test_dir = data_dir + '/' + self.args['test_dir']
        num_of_classes = len(next(os.walk(train_dir))[1])
        return num_of_classes

    def disable_autograd(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def __init__(self, args, data_loader):
        self.args = args
        print(self.args['arch'])
        print(self.args['hidden_units'])
        self.output_size = self.get_output_size()
        self.model, self.input_size = self.define_network()
        # have to disable autograd before adding classifier
        # only train the classifier parameters, feature parameter are frozen
        self.disable_autograd()
    
        classifier = self.define_classifier()
        self.model.classifier = classifier
        self.gpu_enabled = self.activate_cuda_if_available()
        self.data_loader = data_loader

class DataLoader(object):
    def __init__(self, args):
        data_dir = args['datadir']
        self.train_dir = data_dir + '/' + args['train_dir']
        self.valid_dir = data_dir + '/' + args['valid_dir']
        self.test_dir = data_dir + '/' + args['test_dir']

        train_transforms = self.define_train_transform()
        test_transforms = self.define_test_transform()

        self.train_datasets = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        self.test_datasets = datasets.ImageFolder(self.test_dir, transform=test_transforms)

        self.train_loaders = torch.utils.data.DataLoader(self.train_datasets, batch_size=56,shuffle=True)
        self.test_loaders = torch.utils.data.DataLoader(self.test_datasets, batch_size=32)

    def define_test_transform(self):
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],
                                                                   [0.229,0.224,0.225])])
        return test_transforms

    def define_train_transform(self):
        train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomRotation(30),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485,0.456,0.406],
                                                                    [0.229,0.224,0.225])])
        return train_transforms

class CliParser(object):
    def get_input_args(self):
        self.parser.add_argument('datadir', nargs='?', type=str,
                            help='path to images folder')
        self.parser.add_argument('--train_dir', type=str, default='train/',
                            help='child directory of images folder, contain set of train images')
        self.parser.add_argument('--test_dir', type=str, default='test/',
                            help='child directory of images folder, contain set of test images')
        self.parser.add_argument('--valid_dir', type=str, default='valid/',
                            help='child directory of images folder, contain set of validation images')
        self.parser.add_argument('--arch', type=str, default='vgg16',
                            help='chose architecture, available options:\n vgg13, vgg16, alexnet')
        self.parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='define learning rate')
        self.parser.add_argument('--hidden_units', nargs='+', type=int, 
                            help='define hidden_units, this can be a list like "1024 500 200"')
        self.parser.add_argument('--epoch', type=int, default=3,
                            help='define number of epochs to run')
        self.parser.add_argument('--save_dir', type=str, default='checkpoint/',
                            help='directories to save checkpoint')
        self.parser.add_argument('--checkpoint_name', type=str, default='checkpoint.pth',
                            help='file name of the checkpoint')
        self.parser.add_argument('--gpu', action='store_true',
                            help='enable gpu')
        self.parser.add_argument('--debug', action='store_true',
                            help='enable debug')

    
        # convert parser namespace to dict
        self.args = vars(self.parser.parse_args())
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.get_input_args()

if __name__ == "__main__":
    #main()
    parser = CliParser()
    data_loader = DataLoader(parser.args)
    network = Network(parser.args, data_loader)
    if parser.args['debug']:
        print("Command line parameter as below:")
        print(parser.args)
        print("\nNetwork as below:")
        print(network.model)
        print("")
    network.train()
    network.save_checkpoint()
