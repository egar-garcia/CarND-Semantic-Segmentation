**Semantic Segmentation Project**


Self-Driving Car Engineer Nanodegree Program

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).


## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Build the Neural Network
#### Does the project load the pretrained vgg model?
##### The function load_vgg is implemented correctly.

#### Does the project learn the correct features from the images?
##### The function layers is implemented correctly.

#### Does the project optimize the neural network?
##### The function optimize is implemented correctly.

#### Does the project train the neural network?
##### The function train_nn is implemented correctly. The loss of the network should be printed while the network is training.

### Neural Network Training

#### Does the project train the model correctly?
##### On average, the model decreases loss over time.

```
Getting and augmenting training set ...
Training set retrieved
Training...

EPOCH 1 ...
2018-05-18 03:37:22.017313: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.08GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2018-05-18 03:37:22.213145: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.08GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2018-05-18 03:37:23.757216: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.08GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2018-05-18 03:37:23.924253: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.08GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
Avg Loss: 0.428
EPOCH 2 ...
Avg Loss: 0.157
EPOCH 3 ...
Avg Loss: 0.129
EPOCH 4 ...
Avg Loss: 0.122
EPOCH 5 ...
Avg Loss: 0.129
EPOCH 6 ...
Avg Loss: 0.095
EPOCH 7 ...
Avg Loss: 0.087
EPOCH 8 ...
Avg Loss: 0.076
EPOCH 9 ...
Avg Loss: 0.080
EPOCH 10 ...
Avg Loss: 0.069
EPOCH 11 ...
Avg Loss: 0.068
EPOCH 12 ...
Avg Loss: 0.066
EPOCH 13 ...
Avg Loss: 0.062
EPOCH 14 ...

```

#### Does the project use reasonable hyperparameters?
##### The number of epoch and batch size are set to a reasonable number.

#### Does the project correctly label the road?
##### The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

