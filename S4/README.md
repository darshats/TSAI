Part 1
In part 1, we create an excel file that shows forward and back propagation in a simple NN. ![Excel](https://github.com/darshats/TSAI/blob/main/S4/excel.png)
The learning rate and initial weights are configuration, and via macros each of the parameters are updated. There is one row per epoch which takes parameter values updated from previous row, using learning rate and computes new loss value.
For forward pass, the output neurons are computed using sigmoid of the linear combination of weights with inputs. During the backward pass, the derivatives are implemented using Excel formulas, and then the weight updates are made by running those formulas inline.
We can see that as the learning rate increases from 0.1 to 10, the NN converges quickly i.e. the loss monotonically goes to zero. The slope is sharper for higher values of alpha.
