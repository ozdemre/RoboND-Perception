## RoboND-Perception

This is my implementation for Udacity Robotics Software Engineer Nanodegree Program Perception project.

**Description of Files**
1. [YamlFiles](/RoboND-Perception-Project/pr2_robot/scripts/) : Output 1-2-3 .yaml file location.
2. [object_recognition.py](RoboND-Perception/RoboND-Perception-Project/pr2_robot/scripts) Main Code where perception pipeline happens
**Exercise 1 is implemented**

Here, pipeline is:
Statistical Outlier Filter->Voxel Grid Downsampling->Passthrough over x and z axis->RANSAC plane segmentation

Different from exercises I had to tune the parameters like Leaf size and Max Distance. It will be more difficult when I take the challenge world especially for passthrough filtering.


**Exercise 2 is implemented**

Clustering is applied here. I had to play with the Euclidean Clustering a bit. After couple of times accidentally clustering objects together I managed to tune the parameters.

Here is how point cloud looks like after clustering

![alt text][image1]

**Exercise 3 is implemented**

Here we implemented object recognition. Model training is done separately.

My model training parameters:
1. 960 point cloud data (8 object x 120 different angles). 
2. 96 bins for both color and normal histograms.
3. HSV for color histogram.

I suggest to go higher especially for number of point clouds, but I got some satisfactory result and had lack of resources.

Here is my confusion matrix.

![alt text][image2]



**Test World-1 Results**

Success: 3/3    

Confidence Matrix: biscuits:%88 soap:%68 soap2:%94

[Output_1.yaml](/RoboND-Perception-Project/pr2_robot/scripts/output_1.yaml)

![alt text][image3]

![alt text][image4]

**Test World-2 Results**

Success: 5/5    

Confidence Matrix: biscuits:%89 book:%96 soap:%69 soap2:%91 glue:%93

[Output_2.yaml](/RoboND-Perception-Project/pr2_robot/scripts/output_2.yaml)


![alt text][image5]

![alt text][image6]

**Test World-3 Results**

Success: 8/8    

Confidence Matrix: snacks:**%37** biscuits:%95 soap:%79 book:%95 eraser:%82 stick_notes:%92 soap2:%95 glue:%91

[Output_3.yaml](/RoboND-Perception-Project/pr2_robot/scripts/output_3.yaml)


![alt text][image7]

![alt text][image8]

**One additional note**: To generate output.yaml file names automatically, I modified the pick_and_place_project.launch file by adding one extra rosparam which states the world number
This ROS parameter is called inside of the object_recognition.py function to generate output_*.yaml file.

**My Comments**
I think this is the most exciting Nanodegree project so far. The possibilities are unlimited, from pick and place operations to mobile robot localization, and as I have a Intel Realsense RGBD camera to work on this project was very enlightening for me.
I will add the repo that I work on real hardware here in the future, but here is a picture where I implemented pipeline steps up to object recognition.

**For Future**
I will definitely try the challenge world and advanced pick and place operations in my spare time as I will be requested to implement in a real hardware in the future.

![alt text][image9]

  
[image1]: ./readme_images/homogenous_transform.JPG
[image2]: ./readme_images/confusion_matrix.JPG
[image3]: ./readme_images/world1.JPG
[image4]: ./readme_images/world1_log_n.JPG
[image5]: ./readme_images/world2.JPG
[image6]: ./readme_images/world2_log_n.JPG
[image7]: ./readme_images/world3.JPG
[image8]: ./readme_images/world3_log_n.JPG
[image9]: ./readme_images/realsense.jpg