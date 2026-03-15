Folder  "tools"
side functions used for feature extraction

yeast_fea_extractor.py
feature extraction of scattering images of different species
only the folder locations and name need to be changed


This code extract deep features from pretrained CNN, which is known as transfer learning, one can refer:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
for more information.

Notice that we train a kernel SVM for classification instead of using the standard CNN for classification,
this is because the CNN, whether trained from scratch or finetune, performs poorly due to the insufficient of 
data for CNN


################################ implementation ###########################

The code is implemented using Pytorch on a desktop with GPU and Windows with Spyder, other platforms should work similarly
for Python

The Spyder we used comes from Anaconda, and one can install the latest Pytorch through Anaconda as well as the
required packages as necessary
