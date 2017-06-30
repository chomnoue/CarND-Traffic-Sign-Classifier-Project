#**Traffic Sign Recognition** 

## Project Writeup

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

[label_count_in_training_set]: ./examples/label_count_in_training_set.png "Visualization"
[original_image]: ./examples/original_image.png "Orignal Image"
[gray_scale_image]: ./examples/gray_scale_image.png "Gayscale Image"
[12_priority_road]: ./examples/12_priority_road.png "12_priority_road"
[23_Slippery_road]: ./examples/23_Slippery_road.png "23_Slippery_road"
[4_speed_limit_70]: ./examples/4_speed_limit_70.png "4_speed_limit_70"
[13_yield]: ./examples/13_yield.png "13_yield"
[2_speed_limit_50]: ./examples/2_speed_limit_50.png "2_speed_limit_50"
[9_no_passing]: ./examples/9_no_passing.png "9_no_passing"
[14_Stop]: ./examples/14_Stop.png "14_Stop"
[33_Turn_right_ahead]: ./examples/33_Turn_right_ahead.png "33_Turn_right_ahead"
[1_Speed_limit_30]: ./examples/1_Speed_limit_30.png "1_Speed_limit_30"
[35_Ahead_only]: ./examples/35_Ahead_only.png "35_Ahead_only"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the  traffic signs are represented in the training set.

![Label counts in training set][label_count_in_training_set]

As you can see, labels are not equaly represented, with Speed limit 50km/h (label 2) happening most and Speed limit 20km/h (label 0) happening less.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the colours should not be important to distinguish traffic signs. They might even been distracting


Here is an example of a traffic sign image before and after grayscaling.

![Original Image][original_image]

![Gayscale Image][gray_scale_image]

As a last step, I normalized the image data because gradient descent algorithm performs better on normalized data

I tried to generate additional data because to improve my model capabilty to genaralize, using the [Keras image data generator](https://keras.io/preprocessing/image/)
But it did not work for me as the trained model had only 0.8 accuracy on the validation set after more than 100 epochs.
So for now I just removed data augmentation. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Flatten       		| 400        									|
| Fully connected		| 120       									|
| RELU					|												|
| Dropout       		|           									|
| Fully connected		| 84        									|
| RELU					|												|
| Dropout       		|           									|
| Fully connected		| 43        									|
| Softmax				| 43        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with a learning rate of 0.001. The batch size is 128 and I trained for 20 EPOCHS

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.95
* validation set accuracy of 0.95 
* test set accuracy of 0.936

I choose the implement the well known [LeNet architecture](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)
I believed that as it has been successful to classify image data ( deteting characters in MINST data for example), it could also apply to traffic sign detection
The above results shows that I was not wrong as my model exeeds the expected validation set accuracy, and works well on the training set

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![12_priority_road][12_priority_road]  ![23_Slippery_road][23_Slippery_road]     ![4_speed_limit_70][4_speed_limit_70]
![13_yield][13_yield]          ![2_speed_limit_50][2_speed_limit_50]     ![9_no_passing][9_no_passing]
![14_Stop][14_Stop]           ![33_Turn_right_ahead][33_Turn_right_ahead]
![1_Speed_limit_30][1_Speed_limit_30]  ![35_Ahead_only][35_Ahead_only]

I think that the model should be able to classify all theses images since the signs are clearly visible and well centered.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 12_priority_road      | 12_priority_road   							|
| 23_Slippery_road      | 23_Slippery_road   							|
| 4_speed_limit_70      | 1_Speed_limit_30   							|
| 13_yield      		| 13_yield   									|
| 2_speed_limit_50      | 2_speed_limit_50   							|
| 9_no_passing      	| 9_no_passing   								|
| 14_Stop      		    | 14_Stop   									|
| 33_Turn_right_ahead   | 33_Turn_right_ahead   						|
| 1_Speed_limit_30      | 1_Speed_limit_30   							|
| 35_Ahead_only      	| 35_Ahead_only   								|


The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 95%


####3. Describe how certain the model is when predicting on each of the  new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.
 
For the images with correct predictions, the highest probabilities range from 0.989 to ~1, which means that the model is very confident about thes predictions

For the wrorng prediction (4_speed_limit_70), the model has more doubt, predicting a 1_Speed_limit_30 (probability of 0.676). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .6760         		| 1_Speed_limit_30  							| 
| .2724     			| 4_speed_limit_70 								|
| .0021					| 2_speed_limit_50								|
| .0015	      			| 14_Stop					 				    |
| .0006				    | 18_General_caution     						|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


