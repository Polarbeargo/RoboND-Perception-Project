## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[//]: # (Image References)
[image1]: ./image/8.png
[image2]: ./image/9.png
[image3]: ./image/5.png
[image4]: ./image/6.png
[image5]: ./image/10.png
[image6]: ./image/11.png
[image7]: ./image/12.png
[image8]: ./image/13.png
[image9]: ./image/14.png
[image10]: ./image/16.png
[image11]: ./image/17.png
# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the ???pick_place_routine??? rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
* Downsampled PCL cloud using voxel grid filter with leaf size 0.01m  

![Youtube Demo:](https://img.youtube.com/vi/wr3moRdVGfw/0.jpg)
* Applied Pass through Filter  

![Youtube Demo:](https://img.youtube.com/vi/vsgVgHVBH_k/0.jpg)
* Applied RANSAC plane fitting extracted inliers  

![Youtube Demo:](https://img.youtube.com/vi/VhsuVJ-8mQ0/0.jpg)]
* RANSAC plane fitting Extract outlier  

![Youtube Demo:](https://img.youtube.com/vi/WHuAv5UtK60/0.jpg)

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
The k-d tree used in the Euclidian Clustering algorithm to decrease the computational burden of searching for neighboring points. In here by construct k-d tree with(line 224-225):

```
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
```  

then proceed with cluster extraction as follow(line 227-231):

```
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.02)
ec.set_MinClusterSize(50)
ec.set_MaxClusterSize(1500)
ec.set_SearchMethod(tree)
```   

cluster_indices now contains a list of indices for each cluster.    

![Youtube Demo:](https://img.youtube.com/vi/C2lNOTTNEqU/0.jpg)    

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

Modify the for loop in 'capture_features.py' from range(5) to range (100), set 'using_hsv=True' and train with supporting Vector Machine(SVM). The process as bellow:  
Youtube Demo:   

[![Youtube Demo:](https://img.youtube.com/vi/jKKeYojWtIA/0.jpg)](https://www.youtube.com/watch?v=jKKeYojWtIA&t=66s)     

![][image1]
![][image2]
![][image3]
![][image4]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further. 

Change leaf size to 0.005 and passthrough filter along z and x value in project_template.py(line 183-184 and Line 192-193), change for loop in capture_fearture.py to 300 to add new model Feature in training set to 2400.  

![][image5]
Youtube Demo:   

[![Youtube Demo:](https://img.youtube.com/vi/20xtnMYcr-s/0.jpg)](https://www.youtube.com/watch?v=20xtnMYcr-s)   
![][image10]
![][image11]
![][image8]
[YAML1](https://github.com/Polarbeargo/RoboND-Perception-Project/blob/master/output_1.yaml)   
![][image7]
[YAML2](https://github.com/Polarbeargo/RoboND-Perception-Project/blob/master/output_2.yaml)   
![][image9]
[YAML3](https://github.com/Polarbeargo/RoboND-Perception-Project/blob/master/output_3.yaml)
