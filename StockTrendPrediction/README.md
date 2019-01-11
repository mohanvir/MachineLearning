Included files
  1) Google_Stock_Price_Test.csv
  2) Google_Stock_Price_Train.csv
  3) stock_trend_prediction.py
  
Expected libraries and Software
   1) Anaconda and its dependencies
   2) numpy, matplotlib, pandas and keras
   
   

In this project I used a Recurrent Nerual Network to help me predict the flow of Google Stocks. This project was deigned
for me to gain more experience in using libraries and develope a Recurrent Nerual Network with drop regulation. I implemented
the Recurrent Nerual Network using LSTM (long short term memory) to solve the Vanishing Gradient Descent Problem. I decided to
publish the version with 3 LSTM's because this seemed to be the most effective, but in the future this can be tested and
improved. The "fit" function from keras is used to connect the dataset to the RNN and run. Note that this should not take more
than 20 mins using a CPU, so tensorflow installed on a CPU will be fine. Since there is no way to fully control how the RNN
will learn, when graphing the prediction of the month of January at the end, the lines may slightly change but the overall 
shape should be the same.

This project can be used in many other fields. For example the new chapter of Harry Potter could have been written using an RNN
https://www.geek.com/tech/ai-generated-harry-potter-chapter-wins-the-internet-1725679/

Even Google Translate could be using something very similar to an RNN to translate words between languages. In the future I
plan to make a more powerful RNN that can be used as a text editor. Theres plenty of very wonderful text editors to help
developers, but I think it would be cool to have an AI that can help students learn to code. 
