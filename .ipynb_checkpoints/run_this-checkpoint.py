import torch.utils.data as data
import cv2
import sys
import random
from os import listdir
from os.path import join
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from inference.Compiler import *


# Model Imports
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

#torch.nn.Module.dump_patches = True


def resize_img(png_file_path):
        img_rgb = cv2.imread(png_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (224,224), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(224,224,3))
        bg_img[0:224, 0:224,:] = resized
        bg_img /= 255
        bg_img = np.rollaxis(bg_img, 2, 0)  
        return bg_img
    
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


#Models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        if images.shape[0] < 2:
            features = self.linear(features)
            return features
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(embed_size*2, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, hidden):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1).repeat(1,embeddings.shape[1],1), embeddings), 2)
        #packed = pack_padded_sequence(embeddings, 48, batch_first=True) 
        output, hidden = self.lstm(embeddings, hidden)
        outputs = self.linear(output)
        return outputs, hidden
    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

def load_best(model):
    base = 'trained/'
    if model == 1:
        base = 'trained2/'
    lst = listdir(base)
    le = 999
    ld = 999
    prev_e = ''
    prev_d = ''
    for i in lst:
        num = i.split('(')[1]
        num = num.split(')')[0]
        num = float(num)
        if i.startswith('encoder'):
            if le > num:
                le = num
                if prev_e != '':
                    os.remove(base + prev_e)
                prev_e = i
            elif le < num:
                os.remove(base + i)
        elif i.startswith('decoder'):
            if ld > num:
                ld = num
                if prev_d != '':
                    os.remove(base + prev_d)
                prev_d = i
            elif ld < num:
                os.remove(base + i)
    return base + prev_e, base + prev_d


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def load_val_images(data_dir):
    image_filenames =[]
    images = []
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "png":
            image_filenames.append(filename)
    for name in image_filenames:
        image = resize_img(data_dir+name)
        images.append(image)
    return images



def load_n_run(image_file=None, file=True, model=0):
    #encoder = torch.load(os.path.abspath("model_weights/encoder_resnet34_0.061650436371564865.pt"))
    #decoder = torch.load(os.path.abspath("model_weights/decoder_resnet34_0.061650436371564865.pt"))
    me, md = load_best(model)
    print(me, md)
    encoder = torch.load(me)
    decoder = torch.load(md)
    #encoder = torch.load(os.path.abspath("encoder_resnet34_tensor(0.0435).pt"))
    #decoder = torch.load(os.path.abspath("decoder_resnet34_tensor(0.0435).pt"))
    
    # Initialize the function to create the vocabulary 
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary 
    tokenizer.fit_on_texts([load_doc('vocabulary.vocab')])
    
    
    decoded_words = []
    star_text = '<START> '
    hidden = decoder.init_hidden()
    #image = load_val_images('val/')[0]
    if image_file:
        print('image_file:',image_file)
        image = resize_img(image_file)
    else:
        image = load_val_images('test/')[0]
    #image = Variable(torch.FloatTensor([image]))
    image = Variable(torch.FloatTensor([image]))
    predicted = '<START> '
    for di in range(9999):
        sequence = tokenizer.texts_to_sequences([star_text])[0]
        decoder_input = Variable(torch.LongTensor(sequence)).view(1,-1)
        features = encoder(image)
        #print(decoder_input)
        outputs,hidden = decoder(features, decoder_input,hidden)
        topv, topi = outputs.data.topk(1)
        ni = topi[0][0][0]
        word = word_for_id(ni, tokenizer)
        if word is None:
                continue
        predicted += word + ' '
        star_text = word
        #print(predicted)
        if word == '<END>':
                break
    print(predicted)
    
    #select html tag collection for replacement
    compiler = Compiler('default')
    #generate html content
    compiled_website = compiler.compile(predicted.split())
    
    if file:
        with open('output.html', 'w') as f:
            f.write(compiled_website)
    else:
        print(compiled_website)


if __name__ == "__main__":
    # train on 20 images
    #training(100)
    load_n_run(image_file='./test/test.png', file=True)

