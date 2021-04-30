# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./images/visualization01.png "Visualization 1"
[image2]: ./images/visualization02.png "Visualization 2"
[image3]: ./images/grayscale.png "Gray Scale"
[image4]: ./images/03_speed60_32x32x3.jpg "Traffic Sign 1"
[image5]: ./images/12_PriorityRoad_32x32x3.jpg "Traffic Sign 2"
[image6]: ./images/13_yield_32x32x3.jpg "Traffic Sign 3"
[image7]: ./images/28_ChildrenCrossing_32x32x3.jpg "Traffic Sign 4"
[image8]: ./images/35_AheadOnly_32x32x3.jpg "Traffic Sign 5"
[image9]: ./images/top5result.png "Top 5 softmax probabilities for the prediction"



---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is  4410
* The size of test set is 12630
* The shape of a traffic sign image is  (32, 32, 3)
* The number of unique classes/labels in the data set is  43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a example image for each label and bar chart showing examples by class

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is good to reduce the amount of features and execution time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data so that the data has mean zero and equal variance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image |
| Convolution Layer1 | 1x1 stride, same padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		|
| Convolution Layer2	| 1x1 stride, same padding, outputs 10x10x16 |
|        RELU        |                                            |
|    Max pooling     |        2x2 stride,  outputs 5x5x16         |
| Full Connected0 | Flatten, Dropout, output 400 |
| Full Connected1 | Matmul, output 120 |
| RELU,Dropout |  |
| Full Connected2 | Matmul, output 84 |
| RELU,Dropout |  |
| Full Connected3 | Matmul, output 43 |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used learning rate 0.0006, Adam optimizer, batch size 128, number of epochs 200, dropout 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.961
* test set accuracy of 0.953

If a well known architecture was chosen:
* What architecture was chosen? I use LeNet-5. 

* Why did you believe it would be relevant to the traffic sign application? Because it works out well on the image recognition for MNIST.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The accuracy it achived on the training, validation and test sets are all > 0.91 and are its performance is pretty consistence.

  


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The speed limit image might be difficult to classify because it was classied several sign and I add epoch number, then it work well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30kim/h) | Speed limit (20kim/h) |
| Priority road | Priority road |
| Yield	| Yield											|
| Ahead only	| Ahead only	|
| Stop	| Stop      			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.3% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here is result image:

![alt text][image9]

For the first image, the model is relatively sure that this is a "Speed limit(20km/h) sign (probability of 0.6), and the image does contain  other speed limit signs. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .02        | Speed limit(70kim/h) |
| .58     | Speed limit(30kim/h) |
| .39		| Speed limit(20kim/h)	|
| .01	      | Speed limit(120kim/h)	|
| .00				  | End of speed limit(80kim/h) |

For the second image is "Priority road" sign(probability of 1.0)

| Probability |                     Prediction                     |
| :---------: | :------------------------------------------------: |
|     .00     |                Roundabout mandatory                |
|     1.0     |                   Priority road                    |
|     .00     |                     Keep right                     |
|     .00     | End of no passing by vehicles over 3.5 metric tons |
|     .00     |                     Ahead only                     |

For the third image is "Yield" sign(probability of 1.0)

| Probability |   Prediction    |
| :---------: | :-------------: |
|     1.0     |      Yield      |
|     .00     | Turn left ahead |
|     .00     |  Priority road  |
|     .00     |   No passing    |
|     .00     |   Keep right    |

For the fourth image is "Ahed only" sign(probability of 1.0)

| Probability |      Prediction      |
| :---------: | :------------------: |
|     .00     |   Turn left ahead    |
|     .00     | Speed limit(60kim/h) |
|     .00     |    Priority road     |
|     .00     | Go straight or right |
|     1.0     |      Ahead only      |

For the fourth image is "Stop" sign(probability of 1.0) 

| Probability |      Prediction      |
| :---------: | :------------------: |
|     .00     |        Yield         |
|     .00     |   Turn right ahead   |
|     1.0     |         Stop         |
|     .00     | Speed limit(60kim/h) |
|     .00     |      Keep right      |



#### 


