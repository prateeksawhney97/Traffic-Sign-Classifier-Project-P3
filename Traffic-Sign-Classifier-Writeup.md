# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## **Building a Traffic Sign Recognition Project**

## The goals / steps of this project are the following:
#### * Loading the data set
#### * Explore, summarize and visualize the data set
#### * Design, train and test a model architecture
#### * Use the model to make predictions on new images
#### * Analyze the softmax probabilities of the new images
#### * Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/prateeksawhney97/Traffic-Sign-Classifier-Project-P3/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set, by drawing the first image of 35 classes, 43 classes in total.

![download 1](https://user-images.githubusercontent.com/34116562/50345963-d592ba00-0556-11e9-9f11-26364914f745.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution Layer 1   	| Outputs 28x28x6 	|
| RELU					|	Activation applied to output of layer 1	|
| Pooling	      	| Input = 28x28x6, Output = 14x14x6 				|
| Convolution Layer 2	    | Outputs 10x10x16    									|
| RELU		| Activation applied to output of layer 2        									|
| Pooling				| Input = 10x10x16, Output = 5x5x16        									|
|	Flatten					|		Input = 5x5x16, Output =400										|
|		Fully Connected Layer 1				|		Input = 400, Output = 120						|
| RELU		| Activation applied to output of Fully Connected layer 1        									|
|		Fully Connected Layer 2				|		Input = 120, Output = 84						|
| RELU		| Activation applied to output of Fully Connected layer 2        									|
|		Fully Connected Layer 3				|		Input = 84, Output = 43					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used EPOCHS = 20, BATCH_SIZE = 128, rate = 0.001, mu = 0, sigma = 0.1.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

### Here are eight German traffic signs that I found on the web:

![test_1](https://user-images.githubusercontent.com/34116562/50350706-f3b3e680-0565-11e9-8535-89058319ac2b.jpeg)
![test_2](https://user-images.githubusercontent.com/34116562/50350707-f4e51380-0565-11e9-912d-f55cb6f86225.jpeg)
![test_3](https://user-images.githubusercontent.com/34116562/50350709-f6164080-0565-11e9-8fdd-9204a3b37552.jpeg)
![test_4](https://user-images.githubusercontent.com/34116562/50350711-f7e00400-0565-11e9-8836-9678b74452a5.jpeg)
![test_5](https://user-images.githubusercontent.com/34116562/50350713-f9a9c780-0565-11e9-8e98-a2ba70dd4478.jpeg)
![test_6](https://user-images.githubusercontent.com/34116562/50350714-fc0c2180-0565-11e9-9fb9-ae7fa91c2302.jpeg)
![test_7](https://user-images.githubusercontent.com/34116562/50350717-fe6e7b80-0565-11e9-8a18-57957f849787.jpeg)
![test_8](https://user-images.githubusercontent.com/34116562/50350721-01696c00-0566-11e9-99fd-240eecc9b2a4.jpeg)

### The eight German Traffic signs after resizing them to 32x32:

![download 2](https://user-images.githubusercontent.com/34116562/50350874-7d63b400-0566-11e9-9a4e-7ee99104a6ba.png)
![download 3](https://user-images.githubusercontent.com/34116562/50350876-7e94e100-0566-11e9-89e0-c5e67b41d75c.png)
![download 4](https://user-images.githubusercontent.com/34116562/50350877-805ea480-0566-11e9-8ed0-d964bf3b98ca.png)
![download 5](https://user-images.githubusercontent.com/34116562/50350878-82286800-0566-11e9-9677-eebe3aff8a6a.png)
![download 6](https://user-images.githubusercontent.com/34116562/50350880-848ac200-0566-11e9-9a3f-16d2781a2dd7.png)
![download 7](https://user-images.githubusercontent.com/34116562/50350894-8c4a6680-0566-11e9-94b9-fee4c491687e.png)
![download 8](https://user-images.githubusercontent.com/34116562/50350901-8eacc080-0566-11e9-9223-209ae8326efb.png)
![download 9](https://user-images.githubusercontent.com/34116562/50350906-910f1a80-0566-11e9-9f08-4be597a04f5e.png)



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction on the eight german traffic signs:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield    		| Yield 									| 
| Stop     			| Stop 										|
| Stop					| Stop											|
| Pedestrians	      		| Speed limit (50km/h)					 				|
| Road Work			| Road Work 							|
| Keep right			| Yield      							|
| Speed limit (70km/h)			| Bicycles crossing  							|
| No entry			| No entry     							|


The model was able to correctly guess 5 of the 8 traffic signs. Test Accuracy on new loaded images = 0.625.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


