# FaceRecogniser.
# Brandon Joffe
# 2016
#
# Copyright 2016, Brandon Joffe, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code used in this project included opensource software (openface)
# developed by Brandon Amos
# Copyright 2015-2016 Carnegie Mellon University

import cv2
import numpy as np
import os
import glob
import dlib
import sys
import argparse
from PIL import Image
import pickle
import math
import datetime
import threading
import logging
from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import time
from operator import itemgetter
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import atexit
from subprocess import Popen, PIPE
import os.path
import numpy as np
import pandas as pd
import aligndlib
import openface

import torch
import loadOpenFace  # https://github.com/thnkim/OpenFacePytorch

import csv

logger = logging.getLogger(__name__)

start = time.time()
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
alignedImgDir = os.path.join(fileDir, 'aligned-images')
genEmbedDir =  os.path.join(fileDir, 'generated-embeddings')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
args = parser.parse_args()
args.cuda = True

class FaceRecogniser(object):
    """This class implements face recognition using Openface's
    pretrained neural network and a Linear SVM classifier. Functions
    below allow a user to retrain the classifier and make predictions
    on detected faces"""

    def __init__(self):
        #self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
        self.net = loadOpenFace.prepareOpenFace(useCuda=args.cuda, gpuDevice=0, useMultiGPU=False).eval()
        
        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.neuralNetLock = threading.Lock()
        self.predictor = dlib.shape_predictor(args.dlibFacePredictor)

        logger.info("Opening classifier.pkl to load existing known faces db")
        with open("generated-embeddings/classifier.pkl", 'rb') as f: # le = labels, clf = classifier
            (self.le, self.clf) = pickle.load(f, encoding='bytes') # Loads labels and classifier SVM or GMM

    def make_prediction(self,rgbFrame,bb):
        """The function uses the location of a face
        to detect facial landmarks and perform an affine transform
        to align the eyes and nose to the correct positiion.
        The aligned face is passed through the neural net which
        generates 128 measurements which uniquly identify that face.
        These measurements are known as an embedding, and are used
        by the classifier to predict the identity of the person"""

        landmarks = self.align.findLandmarks(rgbFrame, bb)
        if landmarks == None:
            logger.info("///  FACE LANDMARKS COULD NOT BE FOUND  ///")
            return None
        alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            logger.info("///  FACE COULD NOT BE ALIGNED  ///")
            return None

        logger.info("////  FACE ALIGNED  // ")
        with self.neuralNetLock :
            persondict = self.recognize_face(alignedFace)

        if persondict is None:
            logger.info("/////  FACE COULD NOT BE RECOGNIZED  //")
            return persondict, alignedFace
        else:
            logger.info("/////  FACE RECOGNIZED  /// ")
            return persondict, alignedFace

    def recognize_face(self,img):
        rep1 = self.getRep(img) # Gets embedding representation of image
        if rep1 is None:
            return None
        logger.info("Embedding returned. Reshaping the image and flatting it out in a 1 dimension array.")
        #print("rep1", type(rep1), rep1)
        #rep = np.array(rep1).reshape(1, -1)   #take the image and  reshape the image array to a single line instead of 2 dimensionals
        start = time.time()
        logger.info("Submitting array for prediction.")
        #predictions = self.clf.predict_proba(rep1).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
        # Computes probabilities of possible outcomes for samples in classifier(clf).
        predictions = self.clf.predict_proba(rep1.cpu().detach().numpy()).ravel() 
        #logger.info("We need to dig here to know why the probability are not right.")
        maxI = np.argmax(predictions)
        person1 = self.le.inverse_transform([maxI])[0] #TODO check if  return value is always list/array
        confidence1 = int(math.ceil(predictions[maxI]*100))

        logger.info("Recognition took {} seconds.".format(time.time() - start))
        logger.info("Recognized {} with {:.2f} confidence.".format(person1, confidence1))
        print("Recognition took {} seconds.".format(time.time() - start))
        print("Recognized {} with {:.2f} confidence.".format(person1, confidence1))

        persondict = {'name': person1, 'confidence': confidence1, 'rep':rep1}
        return persondict

    def getRep(self, alignedFace):
        bgrImg = alignedFace
        if bgrImg is None:
            logger.error("unable to load image")
            return None
        #print("bgrImg", bgrImg)
        logger.info("Tweaking the face color ")
        img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        #img = bgrImg.getRGB()
        #
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32) / 255.0
        start = time.time()
        logger.info("Getting embedding for the face")
        
        I_ = torch.from_numpy(img).unsqueeze(0)
        if args.cuda:
            I_ = I_.cuda()

        #print("I_", I_)
        rep = self.net.forward(I_) # Gets embedding - 128 measurements
        return rep
    
    def reloadClassifier(self):
        with open("generated-embeddings/classifier.pkl", 'r') as f: # Reloads character stream from pickle file
            (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
        logger.info("reloadClassifier called")
        return True

    def trainClassifier(self):
        """Trainng the classifier begins by aligning any images in the
        training-images directory and putting them into the aligned images
        directory. Each of the aligned face images are passed through the
        neural net and the resultant embeddings along with their
        labels (names of the people) are used to train the classifier
        which is saved to a pickle file as a character stream"""

        logger.info("trainClassifier called")

        path = fileDir + "/aligned-images/cache.t7"
        try:
            os.remove(path) # Remove cache from aligned images folder
        except:
            logger.info("Failed to remove cache.t7. Could be that it did not existed in the first place.")
            pass

        start = time.time()
        aligndlib.alignMain("training-images/","aligned-images/","outerEyesAndNose",args.dlibFacePredictor,args.imgDim)
        logger.info("Aligning images for training took {} seconds.".format(time.time() - start))
        done = False
        start = time.time()

        done = self.generate_representation()

        if done is True:
            logger.info("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
            start = time.time()
            # Train Model
            self.train("generated-embeddings/","LinearSvm",-1)
            logger.info("Training took {} seconds.".format(time.time() - start))
        else:
            logger.info("Generate representation did not return True")

    def generate_representation(self):
        labels = []
        reps = []
        with open(genEmbedDir + os.sep + 'labels.csv', 'w') as labels_file:
            label_writer = csv.writer(labels_file)
            idx = 0
            last_cls = ""
            for subdir, dirs, files in os.walk(alignedImgDir):
                for filename in files:
                    filepath = subdir + os.sep + filename
                    print(filename)
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        print(filepath)
                        cls = subdir.split(os.sep)[-1]
                        if cls != last_cls:
                            idx += 1
                            last_cls = cls
                        #alignedImage = openface.data.Image(cls, filename, filepath)
                        alignedImage = cv2.imread(filepath)
                        #print(alignedImage)
                        rep = self.getRep(alignedImage)
                        #print(rep)
                        label_row = [str(idx), 'aligned-images' + os.sep + cls + os.sep + filename]
                        label_writer.writerow(label_row)
                        reps.append(rep.cpu().detach().numpy())
            
        if reps:
            np.save(genEmbedDir + os.sep + 'reps.npy', np.row_stack(reps))
        return True

    def train(self, workDir, classifier, ldaDim):
        fname = "{}labels.csv".format(workDir) #labels of faces
        logger.info("Loading labels " + fname + " csv size: " +  str(os.path.getsize("{}reps.csv".format(workDir))))
        print("Loading labels " + fname + " csv size: " +  str(os.path.getsize("{}reps.csv".format(workDir))))
        if os.path.getsize(fname) > 0:
            logger.info(fname + " file is not empty")
            print(fname + " file is not empty")
            #labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
            pd_csv = pd.read_csv(fname, header=None).to_numpy()
            print(pd_csv)
            #labels = pd_csv.values[:, 1]
            labels = pd_csv[:, 1]
            logger.info(labels)
        else:
            logger.info(fname + " file is empty")
            print(fname + " file is empty")
            labels = "1:aligned-images/dummy/1.png"  #creating a dummy string to start the process
        #print("1>labels: {}".format(labels))
        logger.debug(map(os.path.dirname, labels))
        logger.debug(map(os.path.split,map(os.path.dirname, labels)))
        logger.debug(map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels))))
       
        labels = list(map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels))))
        #print("2>labels: {}".format(labels))

        fname = "{}reps.csv".format(workDir) # Representations of faces
        fnametest = format(workDir) + "reps.csv"
        logger.info("Loading embedding " + fname + " csv size: " + str(os.path.getsize(fname)))
        if os.path.getsize(fname) > 0:
            logger.info(fname + " file is not empty")
            embeddings = np.load('{}reps.npy'.format(workDir), allow_pickle=True) 
        else:
            logger.info(fname + " file is empty")
            embeddings = np.zeros((2,150)) #creating an empty array since csv is empty
        
        #print("embeddings", embeddings)
        
        print("labels {}".format(labels))
        # LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1
        self.le = LabelEncoder().fit(labels) 
        # Fits labels to model
        labelsNum = self.le.transform(labels)
        nClasses = len(self.le.classes_)
        logger.info("Training for {} classes.".format(nClasses))

        if classifier == 'LinearSvm':
            self.clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier == 'GridSearchSvm':
            print("""
            Warning: In our experiences, using a grid search over SVM hyper-parameters only
            gives marginally better performance than a linear SVM with C=1 and
            is not worth the extra computations of performing a grid search.
            """)
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            self.clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier == 'GMM':
            self.clf = GMM(n_components=nClasses)
        elif classifier == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            self.clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier == 'DecisionTree':  # Doesn't work best
            self.clf = DecisionTreeClassifier(max_depth=20)
        elif classifier == 'GaussianNB':
            self.clf = GaussianNB()
        # ref: https://jessesw.com/Deep-Learning/
        elif classifier == 'DBN':
            from nolearn.dbn import DBN
            self.clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=1)
            
        if ldaDim > 0:
            clf_final =  self.clf
            self.clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                ('clf', clf_final)])

        self.clf.fit(embeddings, labelsNum) #link embeddings to labels

        fName = "{}/classifier.pkl".format(workDir)
        logger.info("Saving classifier to '{}'".format(fName))
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'wb') as f:
            pickle.dump((self.le,  self.clf), f) # Creates character stream and writes to file to use for recognition  
        print("Training finished!")
            
    def getSquaredl2Distance(self,rep1,rep2):
        """Returns number between 0-4, Openface calculated the mean between
        similar faces is 0.99 i.e. returns less than 0.99 if reps both belong
        to the same person"""
        #print("rep1",rep1)
        #print("rep2",rep2)        
        #d = rep1 - rep2
        #d =  np.sum(np.square(rep1.detach().numpy() - rep2.detach().numpy()))
        d = torch.norm(rep1 - rep2, 2, 1).item()
        print("d", d)        
        return d