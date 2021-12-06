This project implements custom resnet in S9model.py
Also used OneCycleLR. FindLR was used to find the max LR for which the loss was minimal before it rises up sharply. Adam optimizer, it was about 0.47. 
Min LR did not work out as max LR/10, best value based on trial-error was maxLR/470 i.e. starting at 0.001. MaxLR/300 or MaxLR/600 caused non-convergence.

Max test accuracy reached at iteration 24 was 86.17
There was jump in initial test accuracy reaching 60% at epoch 4, and at 84% by epoch 19. Unfortunately after that it slowed down.
