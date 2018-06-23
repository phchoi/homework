#!/usr/bin/env python3.5


import os
import argparse
import numpy
import torch
import json
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image

### This is the Image predictor
class ImagePredictor(object):
    def __init__(self, image, model, mapper_file, gpu_enabled):
        '''
        This image predictor takes 4 parameters
        image -- the input image to be checked
        model -- the network to be tested, this should be a trained model
        mapper_file -- the file containing the category label and name
        gpu_enabled -- whether to enable gpu for the check
        '''
        self.output = self.predict(image, model, gpu_enabled)
        self.labels = self.name_mapper(mapper_file)

    def process_image(self, image):
        '''
        this takes the input image and transforms it 
        '''
        img = Image.open(image)
        width = img.size[0]
        height = img.size[1]
        resized_height, resized_width = 0, 0
        cropped_height, cropped_width = 224, 224
        
        if width > height:
            resized_height = 256
            resized_width = width * resized_height / height
        else:
            resized_width = 256
            resized_height = height * resized_width / width
            
        img.thumbnail((resized_width, resized_height),Image.ANTIALIAS)
          
        left = (resized_width - cropped_width)/2
        top = (resized_height - cropped_height)/2
        right = (resized_width + cropped_width)/2
        bottom = (resized_height + cropped_height)/2
    
        cropped_img = img.crop((left, top, right, bottom))
            
        img_in_np = numpy.array(cropped_img)
        img_in_np = img_in_np / 255
        mean = numpy.array([0.485, 0.456, 0.406])
        std = numpy.array([0.229, 0.224, 0.225])
        img_in_np = (img_in_np - mean)/std
        img_in_np = numpy.transpose(img_in_np, (2,0,1))
        img_in_pytorch = torch.from_numpy(img_in_np)
        
        return img_in_pytorch

    def predict(self, image, model, gpu_enabled):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
    
        model.eval()
    
        image_in_pytorch = self.process_image(image)
        if gpu_enabled:
            test_img_in_tensor = image_in_pytorch.cuda().float()
            test_img_in_tensor.cuda()
        else:
            test_img_in_tensor = image_in_pytorch.float()
            test_img_in_tensor.cpu()
    
        test_img_in_tensor = test_img_in_tensor.unsqueeze(0)
    
        output = model.forward(Variable(test_img_in_tensor))
        return torch.topk(torch.exp(output), 5)

    def name_mapper(self, mapper_file):
        '''
        Read in the mapping file to map from category label to category name
        '''
        with open(mapper_file, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
    
## This is the checkpoint loader
class CheckpointLoader(object):
    def __init__(self, args):
        '''
        This checkpoint loader takes 1 dict as parameter
        args - the dict containing below arguments
        args['checkpoint'] - the checkpoint file path
        args['gpu'] - the flag indicating gpu processing or not
        '''
        self.model = self.load_checkpoint(args['checkpoint'])
        self.args = args
        self.gpu_enabled = self.activate_cuda_if_available()

    def activate_cuda_if_available(self):
        '''
        if user enabled gpu and gpu is available, gpu will be used
        otherwise it will use cpu
        '''
        if self.args['gpu'] and torch.cuda.is_available():
            # Move model parameters to the GPU
            self.model.cuda()
            return True
        else:
            self.model.cpu()
            return False

    def load_checkpoint(self, checkpoint_path):
        '''
        load the checkpoint and recreate the network
        '''
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        classifier = self.define_classifier(checkpoint['input_size'],
                                            checkpoint['output_size'],
                                            checkpoint['hidden_layers'])
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def define_classifier(self, input_size, output_size, hidden_layers):
        '''
        define the classifier from the input_size, output_size and hidden_layers
        '''
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
        
## This is the CLI parser
class PredictCliParser(object):
    def __init__(self):
        '''
        create the parser and add the parser flag
        '''
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.get_input_args()

    def get_input_args(self):
        self.parser.add_argument('input', nargs='?', type=str,
                            help='file to be classified')
        self.parser.add_argument('checkpoint', nargs='?', type=str, default='checkpoint/checkpoint.pth',
                            help='file to load previously saved checkpoint')
        self.parser.add_argument('--top_k', type=int, default=5,
                            help='default top_k to show')
        self.parser.add_argument('--category_names', type=str, default='somefile.json',
                            help='file to map categories to real names')
        self.parser.add_argument('--gpu', action='store_true',
                            help='enable gpu')
        self.parser.add_argument('--debug', action='store_true',
                            help='enable debug')


        # convert parser namespace to dict
        self.args = vars(self.parser.parse_args())


### main method
if __name__ == "__main__":
    '''
    main method
    1. it takes and parse the CLI parameter
    2. recreate the network from checkpoint
    3. predict the image class
    4. show the result
    '''
    parser = PredictCliParser()
    network = CheckpointLoader(parser.args).model
    if parser.args['debug']:
        print("Command line parameter as below:")
        print(parser.args)
        print("\nNetwork as below:")
        print(network)
        print("")
    gpu_enabled = CheckpointLoader(parser.args).gpu_enabled
    predicted_results = ImagePredictor(parser.args['input'], network, parser.args['category_names'], gpu_enabled)
    probs, classes = predicted_results.output
    labels = predicted_results.labels
    for idx, probs in zip(classes.cpu().numpy()[0], probs.cpu().detach().numpy()[0]):
        print("Probability of '%s' (class: %s) is %f" % (labels[str(idx)], idx, probs) )
