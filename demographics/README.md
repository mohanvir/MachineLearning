# demographic segumentation is used to predict future cases. 
What is demographics segumentation and why is it useful?
  - This process is used alot in marketing to find which users are most likely to perform a task. 
  - You will need data that is collected properly for this model, in most cases, the more user data(independent variables) 
        will give more accurate results. The dataset will be divided into a training and testing set
  - Given a dataset, an Artificial Neural Network is made that will learn from the data. 
  - At the end, looking at the weights between Nodes and overall characteristics can tell which buyer is most likely to buy
      this can be used for companies to look for certain demographics and target their customers more effectively.
      

In this repository you will find the following files.
  1) main.py
  2) analysis.py
  3) parameterTuning.py
  4) BankingData.csv
  
  1) main.py  
    main.py is the core of this project. A dataset is imported in, and all data preprocessing is done here. Note that the
    the given main.py is used for BankingData.csv, THIS TEMPLATE NEEDS CHANGES based on YOUR csv file. Please remember to
    change the independent variables and dependent variables in the correct locations. 
    The Artificial Neural Network has two hidden layers and more can be added easily.
    
  2) analysis.py
    This file is used to check how well the built model will perform. The model will run 10 times and find calculate the 
    average mean and variance to show how well the model will work. This is neccessary because no model will work 100% of the
    time.
   
   3) parameterTuning.py
      This file is used to check how different parameters can affect the Artifical Neural Network. for example instead of
      using a sigmoid function we decide to use a different function. This is good for testing different batch sizes,
      epochs, optimizers and much more.  NOTE: I recommended if you are going to run this file, try to run it on a GPU, on a 
      CPU this could take hours.
    
    4) BankingData.csv
      This is the given csv which contains user data (not real people). 10,000 individuals data was collected. 10 independent
      variables and 1 dependnet variable. 
