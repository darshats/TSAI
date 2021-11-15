This is assignment S7. Model is kept in a separate file called S7model.py

The model uses does not use max pooling. Stride=2 is used to reduce size by half. In all other places, dilated convolution with kernel=3, dilation=2 is used so effective kernel size is 5.
Last fully connected layer is replaced by a convolution layer. There is a GAP layer before it.

After 174 epochs, test accuracy has crossed 80% but it is moving up extremely slowly. After 300 epochs it has crossed 82%. I will update further if it gets to 85%! !!!

