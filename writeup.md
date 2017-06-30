# **Traffic Sign Recognition** 

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
[12_priority_road]: ./new_images/12_priority_road.png "12_priority_road"
[23_Slippery_road]: ./new_images/23_Slippery_road.png "23_Slippery_road"
[4_speed_limit_70]: ./new_images/4_speed_limit_70.png "4_speed_limit_70"
[13_yield]: ./new_images/13_yield.png "13_yield"
[2_speed_limit_50]: ./new_images/2_speed_limit_50.png "2_speed_limit_50"
[9_no_passing]: ./new_images/9_no_passing.png "9_no_passing"
[14_Stop]: ./new_images/14_Stop.png "14_Stop"
[33_Turn_right_ahead]: ./new_images/33_Turn_right_ahead.png "33_Turn_right_ahead"
[1_Speed_limit_30]: ./new_images/1_Speed_limit_30.png "1_Speed_limit_30"
[35_Ahead_only]: ./new_images/35_Ahead_only.png "35_Ahead_only"


---

### Data Set Summary & Exploration

#### 1. Summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the  traffic signs are represented in the training set.

![Label counts in training set][label_count_in_training_set]

As you can see, labels are not equaly represented, with Speed limit 50km/h (label 2) happening most and Speed limit 20km/h (label 0) happening less.

###Design and Test a Model Architecture

#### 1. Preprocessing the image data. 

As a first step, I decided to convert the images to grayscale because the colours should not be important to distinguish traffic signs. They might even been distracting


Here is an example of a traffic sign image before and after grayscaling.

![Original Image][original_image]

![Gayscale Image][gray_scale_image]

As a last step, I normalized the image data because gradient descent algorithm performs better on normalized data

I tried to generate additional data to improve my model capabilty to genaralize, using the [Keras image data generator](https://keras.io/preprocessing/image/)
But it did not work for me as the trained model had only 0.8 accuracy on the validation set after more than 100 epochs.
So for now I just removed data augmentation. 


#### 2. Final model architecture

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
 


#### 3. Training the model

To train the model, I used an AdamOptimizer with a learning rate of 0.001. The batch size is 128 and I trained for 20 EPOCHS

#### 4. Approach taken for finding a solution

My final model results were:
* training set accuracy of 0.95
* validation set accuracy of 0.95 
* test set accuracy of 0.936

I choose the implement the well known [LeNet architecture](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)
I believed that as it has been successful to classify image data ( deteting characters in MINST data for example), it could also apply to traffic sign detection.
I added a droput the convolution and fully connected layers to reduce overfitting. 
The above results shows that I was not wrong as my model exeeds the expected validation set accuracy, and works well on the test set

 

### Testing The Model on New Images

#### 1. Choosen  German traffic signs images

Here are ten German traffic signs that I found on the web:

![12_priority_road][12_priority_road]  ![23_Slippery_road][23_Slippery_road]     ![4_speed_limit_70][4_speed_limit_70]
![13_yield][13_yield]          ![2_speed_limit_50][2_speed_limit_50]     ![9_no_passing][9_no_passing]
![14_Stop][14_Stop]           ![33_Turn_right_ahead][33_Turn_right_ahead]
![1_Speed_limit_30][1_Speed_limit_30]  ![35_Ahead_only][35_Ahead_only]

I think that the model should be able to classify all theses images since the signs are clearly visible and well centered.

#### 2. Model's predictions on the new traffic signs

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


#### 3. Model certainty for the  new images

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.
 
For the images with correct predictions, the highest probabilities range from 0.989 to ~1, which means that the model is very confident about thes predictions

For the wrorng prediction (4_speed_limit_70), the model has more doubt, predicting a 1_Speed_limit_30 (probability of 0.676). 
The top five soft max probabilities for that image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .6760         		| 1_Speed_limit_30  							| 
| .2724     			| 4_speed_limit_70 								|
| .0021					| 2_speed_limit_50								|
| .0015	      			| 14_Stop					 				    |
| .0006				    | 18_General_caution     						|



