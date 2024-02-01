#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import torchvision
import argparse
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import torch
from Network.Network import BaselineNet, BatchNormNet, ResNet, ResNeXT, DenseNet
from torchvision.datasets import CIFAR10
from Misc.MiscUtils import *
from Misc.DataUtils import *
import seaborn as sns
import distutils.util



# Don't generate pyc codes
sys.dont_write_bytecode = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

def SetupAll():
    """
    Outputs:
    ImageSize - Size of the Image
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]

    return ImageSize

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img
    
def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred, ConfusionMatrixPath):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')

    # Save plot.
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    # Save the plot as an image file.
    cm_path = os.path.join(ConfusionMatrixPath, 'cm.jpg')
    plt.savefig(cm_path)
    plt.close()


def TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelType):
    """
    Inputs: 
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    TestSet - The test dataset
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to /content/data/TxtFiles/PredOut.txt
    """
    # Predict output with forward pass, MiniBatchSize for Test is 1
    # model = CIFAR10Model(InputSize=3*32*32,OutputSize=10).to(DEVICE)
    if ModelType == 'Baseline':
        model = BaselineNet(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'BatchNorm':
        model = BatchNormNet(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'ResNet':
        model = ResNet(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'ResNeXT':
        model = ResNeXT(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'DenseNet':
        model = DenseNet(InputSize=3*32*32,OutputSize=10)

   
    
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
    
    OutSaveT = open(LabelsPathPred, 'w')
    model.eval()
    for count in tqdm(range(len(TestSet))):
        Img, Label = TestSet[count]
        Img, ImgOrg = ReadImages(Img)
        PredT = torch.argmax(model(torch.tensor(Img))).item()

        OutSaveT.write(str(PredT)+'\n')
    OutSaveT.close()

       
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/aa/144model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--SelectTestSet', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help='Choose the set to run the test on, Default:True')
    Parser.add_argument('--ModelType', default='Baseline', help='Model to use for training Model Types are Baseline, BatchNorm, ResNet, ResNeXt, DenseNet, Default:Baseline')
    Parser.add_argument('--ConfusionMatrixPath', default='./Logs', help='Path to save the confusion matrix Default:./Logs')


    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    LabelsPath = Args.LabelsPath
    SelectTestSet = Args.SelectTestSet
    ModelType = Args.ModelType
    ConfusionMatrixPath = Args.ConfusionMatrixPath


    transform = torchvision.transforms.Compose([
        ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])

    TrainSet = CIFAR10(root='data/', train=True, transform=transform, download=True)
    TestSet = CIFAR10(root='data/', train=False, transform=transform, download=True)


    # Setup all needed parameters including file reading
    ImageSize = SetupAll()

    # Define PlaceHolder variables for Predicted output
    LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    if SelectTestSet:
       print("Running on Test Set")
       TestOperation(ImageSize, ModelPath, TestSet, LabelsPathPred, ModelType)
       LabelsPath = './TxtFiles/LabelsTest.txt'
    else:
        TestOperation(ImageSize, ModelPath, TrainSet, LabelsPathPred, ModelType)
        LabelsPath = './TxtFiles/LabelsTrain.txt'


    # Plot Confusion Matrix
    LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    ConfusionMatrix(LabelsTrue, LabelsPred, ConfusionMatrixPath)

     
if __name__ == '__main__':
    main()
 
