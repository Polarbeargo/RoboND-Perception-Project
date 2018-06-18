#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

scene_num = 1 
# Helper function to get surface normals


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages


def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict


def pass_through_filter(point_cloud, axis, axis_min, axis_max):

    # Create a pass_through filter object.
    pass_through = point_cloud.make_passthrough_filter()
    filter_axis = axis
    pass_through.set_filter_field_name(filter_axis)
    pass_through.set_filter_limits(axis_min, axis_max)

    # Return the pass_through point cloud
    return pass_through.filter()


def outlier_filter(point_cloud, neighboring_pts, scale_factor):

    # Create the filter object
    outlier_filter = point_cloud.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(neighboring_pts)

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(scale_factor)

    # Return the filtered data
    return outlier_filter.filter()


def voxel_grid(point_cloud, leaf_size):

    vox = point_cloud.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)

    # Call the filter function to obtain the resultant down sampled point cloud
    return vox.filter()


def ransac_filter(point_cloud, max_distance):

    # Create the segmentation object
    seg = point_cloud.make_segmenter()

    # Set the model used for fitting
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    seg.set_distance_threshold(max_distance)

    # Return the inlier indices and model coefficients of the segment
    return seg.segment()


def euclidean_cluster(white_cloud):
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # Set tolerances for distance threshold
    # As well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(15000)

    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)

    # Extract indices for each of the discovered clusters
    return ec.Extract()

# Helper function to output to yaml file


def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber

def cluster(point_cloud, white_cloud, cluster_indices):
    # Exercise-3
    
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = point_cloud.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)        
        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
 
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
 
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
    return [detected_objects, detected_objects_labels]

def pcl_callback(pcl_msg):

    # Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg) 
    
    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    
    # PassThrough Filter
    
    # 'z' axis' filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    # 'y' axis' filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.45
    axis_max =  0.45
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    # Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # Set threshold scale factor
    x = 1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    
    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficents = seg.segment()
    
    # Extract inliers and outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    cloud_table = cloud_filtered.extract(inliers, negative=False)

# Exercise-2 

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(1500)
    ec.set_SearchMethod(tree)

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_indices = ec.Extract()
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    
    # Cluster point cloud object
    detected_objects, detected_objects_labels = cluster(cloud_objects, white_cloud, cluster_indices)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    centroid = []
    output_yaml = []
    PICK_POSE = Pose()
    PLACE_POSE = Pose()
    test_scene = 3
    TEST_SCENE_NUM = Int32(test_scene)
    OBJECT_NAME = String()
    arm_name = String()

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    place_param = rospy.get_param('/dropbox')
    
    objects = {}
    for i, item in enumerate(object_list_param):    
        if object_list_param[i]['group'] == 'red':
            objects[object_list_param[i]['name']] = 'left'
        elif object_list_param[i]['group'] == 'green':
            objects[object_list_param[i]['name']] = 'right'
    
    # pose of each place group 
    group_pose = {}
    for i, group in enumerate(place_param):
        group_pose[group['name']] = group['position']
    
    # Rotate PR2 in place to capture side tables for the collision map
    # Loop through the pick list
    # initializing messages test_scene_num is constant
    for i, object in enumerate(object_list):
        # 1- object_name
        OBJECT_NAME.data = (object.label).tostring()
        
        # 2- pick_pose 
        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        PICK_POSE.position.x = np.asscalar(centroid[0])
        PICK_POSE.position.y = np.asscalar(centroid[1])
        PICK_POSE.position.z = np.asscalar(centroid[2])
        
        # 3- arm_name
        # Assign the arm to be used for pick_place
        arm_name.data = objects[object.label]
        
        # 4- place_pose
        # Create 'place_pose' for the object
        PLACE_POSE.position.x = group_pose[arm_name.data][0]
        PLACE_POSE.position.y = group_pose[arm_name.data][1]
        PLACE_POSE.position.z = group_pose[arm_name.data][2]
        
        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, arm_name, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        output_yaml.append(yaml_dict)
    send_to_yaml('output_3.yaml', output_yaml)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber(
        "/pr2/world/points", PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher(
        "/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher(
        "/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher(
        "/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher(
        "/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
