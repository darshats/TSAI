# TSAI
Assignment 4 - addition of a digit to an MNIST image via NN

- Data for MNIST is already there, and reused. A random number generator is used to create the second input. A wrapper dataset wraps the two.
  (mnist_image, random_number) --> compound input 
  (mnist label, mnist_label+random_number) --> compound label.
  
- First part of the network is same as that for MNIST. The flattened output of MNIST is concatenated with one-hot encoding of the second input. 
- Loss is calculated separately for mnist and addition label. However since the addition is the complete network (mnist is subset), the add_loss is the one used to minimize
Cross entropy is not really right. If bits fire closer together in the one-hot representation it should lead to lesser loss. So squared loss that actually measures the numerical
difference might be better, but I have run out of time to try that.

Train logs for first 5 epochs:

epoch 0 total_loss 154958.58832985163, mnist correct 5903, add correct 5913

epoch 1 total_loss 154878.0992539525, mnist correct 6042, add correct 5925

epoch 2 total_loss 156330.90835732222, mnist correct 5874, add correct 5728

epoch 3 total_loss 158845.30451714993, mnist correct 6027, add correct 5966

epoch 4 total_loss 158798.56760770082, mnist correct 5937, add correct 5932
