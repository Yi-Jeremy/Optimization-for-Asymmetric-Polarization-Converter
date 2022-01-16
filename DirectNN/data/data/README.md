These three files are training data, validation data and test data. 
Please note that the validation data has been included in the training data, we use the validation_split parameter of Keras to specify the ratio of dividing the validation set during training (0.1 in our code). 
Therefore, the training data has 9000 pieces of data, and the validation data and test data have 1000 pieces of data respectively.
They all play an important role while TPN training. The test data is used to finally test the prediction performance of the network.
