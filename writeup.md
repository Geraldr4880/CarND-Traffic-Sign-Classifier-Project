# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/stop.jpg "Traffic Sign 1"
[image5]: ./examples/yield.jpg "Traffic Sign 2"
[image6]: ./examples/bumpy.jpg "Traffic Sign 3"
[image7]: ./examples/speedlimit100.png "Traffic Sign 4"
[image8]: ./examples/rightofwayatintersection.png "Traffic Sign 5"
[image9]: ./examples/histograms.png "Histograms"
[image10]: ./examples/training_traffic_signs.png "Training Traffic Signs"
[image11]: ./examples/traffic_signs_before_and_after_preprocessing.png "Traffic Sign Pre-Processing, before and after"
[image12]: ./examples/german_traffic_signs.png "German Traffic Signs, resized RGB original and Pre-Processed gray scale"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Geraldr4880/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the shape command to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows three histograms of the traffic signs within the dataset.

![alt text][image9]

Some RGB images from the training set are shown below:

![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to run an adaptive histogram equalization which at the same time converts the images to grayscale. This improved image contrast and reduced the influence of lighting on the result.

Following the histogram equalization I ran a normalization based on (img-mean(img))/(max(img)-min(img)). This normalization centers the pixel value range around an average value of zero. The pixel value range can be between -1 and 1.
Here is an example of a traffic sign image before and after pre-processing (gray scaling, histogram equalization and normalization).

![alt text][image11]

The normalization makes the data easier to handle for the optimizer, improves numeric stability and hence speed.

I decided to not generate additional data because the required accuracy could be achieved and generating additional data would have generated significant effort either in terms of creating a new data structure (dictionary) in order to sort out the underrepresented traffic signs and manipulating them with suitable image transformation or by using tensor flow augmentation strategies within the session. 
https://towardsdatascience.com/tensorflow-image-augmentation-on-gpu-bf0eaac4c967

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the LeNet architecture with dropout added to the fully connected layers (not the output layer).
I changed the layer sizes slightly (more by intuition than by knowing).

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU 					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | output 10x10x16								|
| RELU 					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flatten		      	| output 400									|
| Fully connected		| output 129  									|
| Relu					| 												|
| Drop out				| 0.8 keep probability							|
| Fully connected		| output 86										|
| Relu					| 			  									|
| Drop out				| 0.8 keep probability							|
| Fully connected		| output 43 									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer.
The training rate was set to 0.001.
I chose softmax cross entropy as loss function.
I ran batches of 512 with 500 epochs. The epoch number was higher than necessary. The results improved only marginally after about 100 epochs.
The required validationa accuracy of 93% was achieved after 20 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.0% 
* test set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture and added drop out to preven overfitting the training data.
* What were some problems with the initial architecture?
It mostly worked fine after some parameter changes.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added drop out to the fully connected layers (not the output layer) in order to prevent overfitting.
* Which parameters were tuned? How were they adjusted and why?
I tuned the convolution layer kernel sizes by trial and error. I also changed the fully connected layers to go from 129 output to 86 and finally 43. That seemed intuitive.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layers work well to identify the basic shapes of the traffic signs such as edges and characteristic shapes.
If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
Mostly because it was provided within the course. I decided to use it as a basis and make changes if needed.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training accuracy is a bit too high. This makes me guess that there still is some amount of over-fitting. However, validation and test set accuracy are within spec. Therefore I decided to not take additional measures.
If I spent more time on the task, I would first use data augmentation and then change the network structure addin another convolutional layer and applying drop out to the conv layers as well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image12] 

The third and fourth image might be difficult to classify because the traffic signs is not within the image center, is quite distorted, has another sign underneath (bumpy road sign) and has unlegible numbers (100km/h) speed limit.
The problems could most likely be solved by more thourough image resizing (less distortion) and applying a proper bounding box (centering).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Bumpy Road    		| Priority road 								|
| Yield					| Yield											|
| 100 km/h	      		| 60km/h						 				|
| Right-of-way at 		| Right-of-way       							|
| the next intersection | the next intersection      					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This cis below the accuracy achieved on the better pre-processed test set (better centering and less distortion).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The stop, yield and right of way at next intersection signs were identified with a rounded certainty of 1.00.
The bumpy road was misidentified with a certainty of 98%. The second best guess would have been a Roundabout Mandatory sign. The third best guess with a likelihood of smaller 1% is actually Bumpy Road.
The 100km/h speed limit was misidentified with a certainty of 89%. The correct speed limit is not among the top 5 guesses. Other likelihoods were 7% for Road Work, 3% for Ahead Only and about 1% for Go Straight or Right.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


