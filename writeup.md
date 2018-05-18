**Semantic Segmentation Project**


Self-Driving Car Engineer Nanodegree Program

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image References)

[image1]: ./runs/1526621995.0750496/um_000011.png "Image example 1"
[image2]: ./runs/1526621995.0750496/um_000013.png "Image example 2"
[image3]: ./runs/1526621995.0750496/um_000022.png "Image example 3"
[image4]: ./runs/1526621995.0750496/umm_000001.png "Image example 4"
[image5]: ./runs/1526621995.0750496/umm_000078.png "Image example 5"
[image6]: ./runs/1526621995.0750496/uu_000048.png "Image example 6"
[image7]: ./runs/1526621995.0750496/uu_000096.png "Image example 7"
[test_video]: ./test_video.mp4 "Test Video"
[labeled_video]: ./result.mp4 "Labeled Video"


## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Build the Neural Network
#### Does the project load the pretrained vgg model?
##### The function load_vgg is implemented correctly.

Please check lines 26-50 of ```main.py```, with my implementation of ```load_vgg()```.

#### Does the project learn the correct features from the images?
##### The function layers is implemented correctly.

Please check lines 53-91 of ```main.py```, with my implementation of ```layers()``` and other helper methods.

#### Does the project optimize the neural network?
##### The function optimize is implemented correctly.

Please check lines 94-114 of ```main.py```, with my implementation of ```optimize()```.

#### Does the project train the neural network?
##### The function train_nn is implemented correctly. The loss of the network should be printed while the network is training.

Please check lines 117-146 of ```main.py```, with my implementation of ```train_nn()```. Below in the next point I include an example of the messages printed when the NN is training.

### Neural Network Training

#### Does the project train the model correctly?
##### On average, the model decreases loss over time.

Here is one example of trainning in 20 ephocs, in general the average loss tends to decrease.

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
EPOCH 15 ...
Avg Loss: 0.054
EPOCH 16 ...
Avg Loss: 0.050
EPOCH 17 ...
Avg Loss: 0.057
EPOCH 18 ...
Avg Loss: 0.075
EPOCH 19 ...
Avg Loss: 0.057
EPOCH 20 ...
Avg Loss: 0.059
Model has been saved in: ./model/model
Training Finished. Saving test images to: ./runs/1526621995.0750496
```

#### Does the project use reasonable hyperparameters?
##### The number of epoch and batch size are set to a reasonable number.

For this project I chose a number of 20 ephocs, after some tests, it was observed that for greater numbers the NN was not doing significant improvement on learning.

In terms of the batch size I picked 2, this was due to memory limitations since my GPU was not working for greater values (preloaded VGG model is already too big).


#### Does the project correctly label the road?
##### The project labels most pixels of roads close to the best solution. The model doesn't have to predict correctly all the images, just most of them. A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road.

These are some images to show how the labeling works:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]


## Extras

### Augmenting images of training sample

Please check lines 150-192 of ```main.py```, with my implementation of the funtion ```get_augmented_batch_function()```, which returns the generator to get the image batches. It includes additional images created from the original set by flipping.

### Applying the trained model to a video

Please check the file ```video.py``` which implements the application of the trained model (saved in the dir ```model```) to a video. It receives as arguments the files for the input and output videos (output video would be labeled). 

P.e. A test video is in the file ```test_video.mp4```, here is a [link to the test video](./test_video.mp4). The labeled video is in the file ```result.mp4```, here is a [link to the labeled video](./result.mp4). 

### Notes

Trained model files were not include in the repository due to their big size, summing around 2.2 GB.
