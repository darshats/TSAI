# TSAI
Assignment 4 - addition of a digit to an MNIST image via NN

- Data for MNIST is already there, and reused. A random number generator is used to create the second input. A wrapper dataset wraps the two.
  (mnist_image, random_number) --> compound input 
  (mnist label, mnist_label+random_number) --> compound label.
  
- First part of the network is same as that for MNIST. The flattened output of MNIST is concatenated with one-hot encoding of the second input. 
- Loss is calculated separately for mnist and addition label. However since the addition is the complete network (mnist is subset), the add_loss is the one used to minimize
Cross entropy is not really right. If bits fire closer together in the one-hot representation it should lead to lesser loss. So squared loss that actually measures the numerical
difference might be better, but I have run out of time to try that.

Train logs for first 3 epochs:
epoch 0 total_loss 151880.3483891487, mnist correct 6043, add correct 5910
epoch 1 total_loss 153105.35919749737, mnist correct 6072, add correct 5956
epoch 2 total_loss 153180.87634515762, mnist correct 6097, add correct 5939
