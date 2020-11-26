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


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(10)
    outlier_filter.set_std_dev_mul_thresh(1)
    cloud = outlier_filter.filter()

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.003
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter for z axis. Ground point clouds are removed
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # PassThrough Filter for x axis. Dropbox clouds are removed
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.003
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)  # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(150)
    ec.set_MaxClusterSize(50000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
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


    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    confidence = []
    print('Number of Clusters = ', len(cluster_indices))
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)

        # Compute the associated feature vector
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))

        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        prediction_confidence_list = clf.predict_proba(scaler.transform(feature.reshape(1, -1)))
        max_conf = np.max(prediction_confidence_list) * 100

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
        print('Detected object', label, 'with', np.round(max_conf), 'percent confidence')

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

    # TODO: Threshold for prediction confidence will be added i the future

    # Now here we call the pr2_mover function to just output the .yaml files. In future I will take the
    # pick and place challenge and modify further
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    object_list_param = rospy.get_param('/object_list')
    test_scene_param = rospy.get_param('/world_id')

    # Initialize variables
    # world_id.data = 1 #used to do this manually,
    # now world_id param is defined in the .launch file and we call it above

    world_id = std_msgs.msg.Int32()
    world_id.data = test_scene_param
    object_name = std_msgs.msg.String()
    arm = std_msgs.msg.String()  # green = right, red = left
    pick_p = geometry_msgs.msg.Pose()
    place_p = geometry_msgs.msg.Pose()
    centroid = []
    counter = 0
    output_yaml = []

    rospy.loginfo('Generating yaml file, however pick and place service will not be requested!')

    # Parse parameters into individual variables
    for target in object_list_param:
        object_name.data = object_list_param[counter]['name']
        found = False
        # Get the PointCloud for a given object and obtain it's centroid
        offset = 0
        for detected in object_list:
            if (object_name.data == detected.label):
                points_arr = ros_to_pcl(detected.cloud).to_array()
                centroids = (np.mean(points_arr, axis=0)[:3])
                found = True

        if (found):

            # Place pose for the object
            pick_p.position.x = float(centroids[0])
            pick_p.position.y = float(centroids[1])
            pick_p.position.z = float(centroids[2])

            # Assign arm and place pose for object
            # this is just a simple trick to avoid stacking. Not guaranteed to work in each case
            place_p.position.x = 0.0 - (offset * 0.05)
            place_p.position.z = 0.8

            if (object_list_param[counter]['group'] == "red"):
                arm.data = "left"
                place_p.position.y = 0.71
            else:
                arm.data = "right"
                place_p.position.y = -0.71

            # Create a list of dictionaries
            yaml_dict = make_yaml_dict(world_id, arm, object_name, pick_p, place_p)
            output_yaml.append(yaml_dict)
            offset += 1

    # Pick and Place Service Request is removed due to the lack of computer resources.
        counter += 1

    # Output your request parameters into output yaml file
    send_to_yaml("output_" + str(world_id.data) + ".yaml", output_yaml)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=20)
    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=20)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=20)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=20)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=20)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=20)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
