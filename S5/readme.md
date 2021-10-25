S5 assignment, new target 99.4% with less than 10k params and 15 iterations

Step 1
The first step sets up the skeleton. However it showed overfitting. Later I also found via RF that at output RF was not 28. It did achieve 99% train and 98% t4est accuracy so basic structure was ok.

Step 2
In step 2 added dropout of 0.1 to reduce overfitting


Step 3
Looking at RF calculations, removed a second maxpool and added more layers. To compensate for increased params, I reduced the max filters to 16 (1-->8-->12-->16-->10). Since test and train accuracy were well matched, reduced dropout to 0.01. 

The is close to what was asked. 8086 parameters and test accuracy consistently at 99.2. It didnt meet the 99.4 criteria but ran out of time. I did not get a chance to check optimizer variations.
