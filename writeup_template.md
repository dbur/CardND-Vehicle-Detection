##Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1.png
[image2]: ./output_images/1.png
[image3]: ./test_images/test6.jpg
[image4]: ./output_images/hogs/test6.jpg
[image5]: ./output_images/all_windows/test6.jpg
[image6]: ./output_images/heatmaps/test6.jpg
[image7]: ./output_images/detected/test6.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Training was contained to the train_model function (lines 227-273 car_detection.py).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

After experimenting with multiple parameters, I was able to consistently achieve an accuracy of 0.99 by converting the image to HSV, using all the channels for HOG features, using a (32,32) spacial feature vector, and using a 32-bin color histogram feature vector. The features were extracted for each image, unraveled, and normalized using sklearn's StandardScaler. The sklearn linear support vector machine model was trained on for 5000 car and non car samples. The resulting model and scaler were pickled for later use.

####2. Explain how you settled on your final choice of HOG parameters.

The best results came out of using 9 orientation bins, 8 pixels per cell, 2 cells per block, and considering all of the color channels. Variations from these parameters would change the runtime for training the model, but would not consistently return a high accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As explained above, the HOG features, color histogram features, and spacial features were all used for training. All were concatenated into a 1d vector and used for training.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At first I had a slide_window and a search_windows function that would be used to iterate over the image to extract the features and use the model to predict whether a car existed in each window. This was later replaced by find_cars (lines 161-224), which performs everything in one place. This function takes as input scale, which works off of the (64,64) shape that the model was trained on, and expands the image based on a scaler to extract the hog features. This allowed the pipeline to extract the hog features once per scale rather than for each window, which caused a large increase in performance. 

The scales used for the test images were 1x to 2x in 7 steps. Ideally this would be limited to certain horizons, but because of implementation, each window scale was scanned across the whole image. This meant that dealing with the heatmap threshold later on would need to be a little higher than the Udacity lessons had.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is the pipeline for test6.png. First is the image itself. Next are the HOG features that would be scanned over. After that are the positive results from inputting each window's feature vectors into the model. Then the heatmap is generated with a threshold of 8, so if at least 8 windows are overlapping the pixels, it will show in the image. Finally, the heatmap's continuous boxes are labeled and the resulting bounding boxes should identify any cars.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline is very similar to the image pipeline, except there is some consideration of frame history. The process_video_frames method (lines 305-341) sets the parameters to what was described above, and loads the pickled model and normalizer for use. The image is converted to BGR in order to later have it's features extracted in HSV. To improve the speed of generating the video, only three scales were used, 1x, 1.5x, and 2x. Heatmaps were generated from the combination of the predictions on those windows and stored in Car_Frame object (lines 351-363).

The Car_Frames object stores the heatmap history, and continuously uses the last 10 heatmaps to created a new summed array. The threshold on this summed array is 20, which is done in an effort to filter out false positives that may pop on screen for one or two frames. This suggests that any bounding boxes that are showed on video are due to persistence.

Another method of filtering used was to only consider some hardcoded values for x and y which define where the midroad barrier is and the horizon is respectively. This limits the window search area.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline still  fails on shadows for some test images. This is translated in the video for very few frames also. In order to combat this, perhaps more than one color channel combination should be used in the feature vector. For example, in addition to HSV, using RGB may be interesting to add in. Additionally, this is nowhere near real-time speed. Even with the single hog pass, the project video took about 30 minutes to generate. Would be nice to find a way to speed that up even more. 

