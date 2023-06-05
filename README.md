# SVD-PINNs
Codes for SVD-PINNs, a transfer learning method based on SVD, experiments on 10-d Allen-Cahn Equation


### Some instructions for implementation

##### 1. We recommend to include learning rate decaying strategy in training

##### 2. Optimizers for singular values should not be companied with momentum, and we experimentally find that simple gradient descent is preferred. But for all other weights except singular values, you may use Adam (all hyper-parameters to be tuned to further improve the performances, this is the "magic" of deep learning).

#### 3. The code is only for experimental comparisons, we are sorry if the code writing is unfriendly to you.

#### 4.We would like to further include the motivation of our method. The Galerkin's idea is to approximate the solution by a linear combinition of some base functions, i.e., $u = \sum_{i=1}^{n} a_i \phi_i(x)$, where $a_i$ are weights and $\phi_{i}(x)$ are base functions. We may regard the second layer of the neural network (2-hidden layer NN) as base functions and  $W_2$ and $b_2$ as corresponding weights. Therefore, the naive (simple) transfer learning method is that all weights are frozen except $W_2$ and $b_2$. However, the capacity of base functions determine the performance/expressive power of the model, so we hope to further improve the base functions but lower the storage compare with the standard PINNs. The intuitive idea is to apply SVD to the lagr matrix $W_1$ and freeze its singular vectors. So all trainable parameters are of size $O(m)$ since the dimension $r, d << m$, while the standard PINNs take costs $O(m^2)$. For more details, please refer to our paper and the latest and updated <a>slides.
