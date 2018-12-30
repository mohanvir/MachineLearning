Please view this using the "Raw" display.


Image recognition is a very powerful tool used to distinguish between two types of pictures. This is used in the medical field 
to diagnose tumors, broken bones and much more. 

  How to run:
   1) libraries
      a) keras
      b) numpy
   2)software
      a) Anaconda with all its dependencies
   3)files provided
      a) ImageRecognition.py this is the file that will train a convolusional neurla network. This script has a relevantly low
           parameter for pixels expecting only a 64 by 64 image. This can be manually changed but note that more pixels will
           increase the run time significantly. Note the fit_generator method was changed, so depending on the version 
           installed, steps_per_epoch and validation_steps must be divided by the batch size.
       b) testingPicture.py this file is very straight forward, make sure to change the test_image.
   4) files not provided
       a)dataset with pictures, I recommend a minimum of 5,000 pictures of each case.
           Example Tumor vs no Tumor : insert 5,000 pictures of tumors, and 5,000 pictures without tumors. 
           A large dataset is required, having too few pictures will cause the learning to not be done properly.
           I recommend having tensor flow installed on your GPU. If using higher pixels and more images and   
           more hidden layers for accuracy, a CPU could take hours, days or even months. Becareful not to hurt your CPU. 
            
    
Details about project and goals:
In this project I explored the power of Convolutional Neural Networks which are a supervised learning method. A convolutional
Neural Network takes in an image and will output a highly educated guess on what the image is, once it is fully trained. The keras libray was used extensively to help. The convolutional layer has 32 feature maps of size 3 by 3. The rectifer function was applied to ensure that non linearity was kept. From there maxppoling was applied, this helped shrink the image size without loosing any key features and if features were offset, they would still be caught. Next, The pooled Feature Maps were flattened into a 1D array that could be passed into an ANN. The last step was to fully connect all layers so back propogation can go back to the random feature detector and change that as well. 

My continuation of this project is to be able to pass in 100s of images and have a CNN that can train and identify on these.
