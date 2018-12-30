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
            
    
