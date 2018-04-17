import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from threading import Lock

import numpy as np
from scipy.linalg import lstsq

from gpd.msg import CloudIndexed
from std_msgs.msg import Header, Int64
from geometry_msgs.msg import Point

from gpd.msg import GraspConfigList

import csv # Remove once passing tf properly
import tf

from mps_msgs.msg import *

cloud = [] # global variable to store the point cloud
combinedList = [] # global variable to store the point cloud when fed from mps_vision
pointRegion = [] # global variable to store the region of each point
mutex = Lock()

def cloudCallback(msg):
    with mutex:
        global cloud
        if len(cloud) == 0:
            for p in point_cloud2.read_points(msg):
                cloud.append([p[0], p[1], p[2]])
        else:
            print("cloud not empty") # NEW

def segmentedCloudCallback(msg):
    with mutex:
        global combinedList
        global pointRegion
        testCloud = msg.regions
        # print(testCloud)
        # rospy.sleep(10)
        for region in range(len(testCloud)):
            for pt in range(len(testCloud[region].points)):
                tempPoint = [testCloud[region].points[pt].x, testCloud[region].points[pt].y, testCloud[region].points[pt].z] 
                combinedList.append(tempPoint)
                pointRegion.append(region)
        # for p in mps_msgs.msg.Perception.read_points(msg, field_names=("x","y,","z"),skip_nans=True):
        #     print("test")
        #     combinedList.append(p)

def callback(msg):
    print("test")
    global grasps
    grasps = msg.grasps

def create_occupancy_map(np_cloud):
    # listener = tf.TransofrmListener()
    # while not rospy.is_shutdown():
    #     try:
    #         (trans, rot) = listener.lookupTransform("table_surface", )
    
    # Read in transfer function # TO DO: do this better...
    try:
        with open("/home/messingj/Documents/catkin_ws/src/mps_vision/logs/tf.txt","r") as file:
            reader = csv.reader(file, delimiter=',')
            ros_tf = [data for data in reader]
        ros_tf = np.asarray(ros_tf, dtype=float)
        threshold = -0.03 # Set this here because it can be smaller if tf is given
    except: # If no transformation matrix is availble, assume camera was directly above table, looking straight down
        print("No tf function found.")
        ros_tf = np.asarray([[0, -1, 0, 0],[-1, 0, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])
        threshold = -0.1
        # return np_cloud

    # Rotate points back to original frame
    np_cloud = np.transpose(np.dot(np.linalg.inv(ros_tf[0:3,0:3]),np.transpose(np_cloud - np.transpose(ros_tf[0:3,3]))))

    # Equation of plane fit to the data (represents table surface)
    A = np.c_[np_cloud[:,0], np_cloud[:,1], np.ones(np_cloud.shape[0])]
    C, _, _, _ = lstsq(A, np_cloud[:,2])
    a, b, c, d = C[0], C[1], -1., C[2]

    # Add plane to point cloud to prevent grasping from under the table
    num_table_pts = 100
    x_table = np.reshape(np.linspace(np.amin(np_cloud[:,0]),np.amax(np_cloud[:,0]),num_table_pts),(num_table_pts,1))
    y_table = np.reshape(np.linspace(np.amin(np_cloud[:,1]),np.amax(np_cloud[:,1]),num_table_pts),(num_table_pts,1))
    xx, yy = np.meshgrid(x_table, y_table)
    z_table = -(a*xx + b*yy + d)/c
    table_points = np.concatenate((np.reshape(xx,(num_table_pts**2,1)), np.reshape(yy,(num_table_pts**2,1)), np.reshape(z_table,(num_table_pts**2,1))), axis=1)

    # Add uncertain points as points in cloud
    z_avg = np.mean(z_table)
    pts_above_table = np_cloud[np_cloud[:,2]<(z_avg-threshold)]
    num_fill = min(10000, (int)(0.1*len(pts_above_table)))
    fill_indices = np.random.randint(len(pts_above_table), size=num_fill) 
    spacing = 0.03 
    for i in fill_indices:
        z_fill = np.arange(pts_above_table[i,2], z_avg-threshold, spacing)
        new_pts = np.tile([pts_above_table[i,0], pts_above_table[i,1], 0], (len(z_fill),1))
        new_pts[:,2] = z_fill
        np_cloud = np.concatenate([np_cloud, new_pts])

    # np_cloud = np.concatenate([np_cloud,table_points]) # Add plane representing table to point point cloud

    # Rotate points back to table frame
    np_cloud = ros_tf[0:3, 0:3].dot(np_cloud.transpose()).transpose() + ros_tf[0:3, 3].transpose()

    # Write new points to file for debugging purposes
    with open("/home/messingj/Documents/catkin_ws/src/mps_vision/logs/test_addedPts.txt","w+") as test_file:
        for i in np_cloud:
            test_file.write(str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n")

    return np_cloud

# ---------------------
# TO DO: Move to Main, but be careful about global variables
# ---------------------

# Create a ROS node.
rospy.init_node('select_grasp')

# Create a ROS node for receiving the tf
# rospy.init_node('tf_listener')
# listener = tf.TransofrmListener()
# rate = rospy.Rate(10)
# while not rospy.is_shutdown():
#     try:
#         (trans,rot) = listener.lookupTransform('/','/table_surface',rospy.Time(0))


# Subscribe to the ROS topic that contains the point cloud, either from the .pcd file or from the mps_vision code.
load_from_file = False
if load_from_file:
    print("loading .pcd file")
    cloud_sub = rospy.Subscriber('/cloud_pcd', PointCloud2, cloudCallback)

    # Do not move on until cloud is filled
    while True:
        with mutex:
            if len(cloud) != 0:
                np_cloud = np.asarray(cloud)
                break
            else:
                rospy.sleep(0.01)
else:
    print("waiting to receive point cloud data")
    cloud_sub = rospy.Subscriber('/segmented_pc', mps_msgs.msg.Perception, segmentedCloudCallback)
    while True:
        with mutex:
            if len(combinedList) != 0:
                np_cloud = np.asarray(combinedList)
                break
            else:
                rospy.sleep(0.01)

print("point cloud loaded")

# Extract the nonplanar indices. Uses a least squares fit AX = b. Plane equation: z = ax + by + c.
# np_cloud = np.asarray(cloud)
X = np_cloud
A = np.c_[X[:,0], X[:,1], np.ones(X.shape[0])]
C, _, _, _ = lstsq(A, X[:,2])
a, b, c, d = C[0], C[1], -1., C[2] # coefficients of the form: a*x + b*y + c*z + d = 0.
dist = ((a*X[:,0] + b*X[:,1] + d) - X[:,2])**2
err = dist.sum()
only_above_table = True # NEW
if only_above_table:
    dist = -np.sign((a*X[:,0] + b*X[:,1] + d) - X[:,2])*dist # Not squaring data and only looking at points above the table.
else:
    pass
idx = np.where(dist > 0.001) 

# print("creating occupancy map")
# np_cloud_mod = np_cloud
np_cloud_mod = create_occupancy_map(np_cloud)
# print(np_cloud_mod.shape)

# Publish point cloud and nonplanar indices.
pub = rospy.Publisher('cloud_indexed', CloudIndexed, queue_size=1)

msg = CloudIndexed()
header = Header()
header.frame_id = "/base_link"
header.stamp = rospy.Time.now()
msg.cloud_sources.cloud = point_cloud2.create_cloud_xyz32(header, np_cloud_mod.tolist())
msg.cloud_sources.view_points.append(Point(0,0,0))
for i in range(np_cloud.shape[0]):
    msg.cloud_sources.camera_source.append(Int64(0))
    msg.cloud_sources.regions.append(Int64(pointRegion[i]))
for i in range(np_cloud.shape[0], np_cloud_mod.shape[0]):
    msg.cloud_sources.camera_source.append(Int64(-1))
    msg.cloud_sources.regions.append(Int64(-1))
for i in idx[0]:
    msg.indices.append(Int64(i))    
s = raw_input('Hit [ENTER] to publish')
pub.publish(msg)
rospy.sleep(2)
print 'Published cloud with', len(msg.indices), 'indices'


# Select a grasp for the robot to execute.
grasps = [] # global variable to store grasps
    
# Subscribe to the ROS topic that contains the grasps.
grasps_sub = rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, callback)

# Wait for grasps to arrive.
rate = rospy.Rate(1)

while not rospy.is_shutdown():    
    if len(grasps) > 0:
        rospy.loginfo('Received %d grasps.', len(grasps))
        break

grasp = grasps[0] # grasps are sorted in descending order by score
print 'Selected grasp with score:', grasp.score

print grasp

