import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from threading import Lock

cloud = [] # global variable to store the point cloud
mutex = Lock()

def cloudCallback(msg):
    with mutex:
        global cloud
        if len(cloud) == 0:
            for p in point_cloud2.read_points(msg):
                cloud.append([p[0], p[1], p[2]])
            # print("cloud shape ", len(cloud)) # NEW
        else:
            print("cloud not empty") # NEW

# Create a ROS node.
rospy.init_node('select_grasp')

# Subscribe to the ROS topic that contains the grasps.
cloud_sub = rospy.Subscriber('/cloud_pcd', PointCloud2, cloudCallback)

# Do not move on until cloud is filled
while True:
    with mutex:
        if len(cloud) != 0:
            break
        else:
            rospy.sleep(0.01)

# Extract the nonplanar indices. Uses a least squares fit AX = b. Plane equation: z = ax + by + c.
import numpy as np
from scipy.linalg import lstsq

np_cloud = np.asarray(cloud) # rename this
X = np_cloud
A = np.c_[X[:,0], X[:,1], np.ones(X.shape[0])]
C, _, _, _ = lstsq(A, X[:,2])
a, b, c, d = C[0], C[1], -1., C[2] # coefficients of the form: a*x + b*y + c*z + d = 0.
dist = ((a*X[:,0] + b*X[:,1] + d) - X[:,2])**2
err = dist.sum()
only_above_table = True # NEW
if only_above_table:
    dist = -np.sign((a*X[:,0] + b*X[:,1] + d) - X[:,2])*dist # NEW. Not squaring data to only look at points above the table.
else:
    pass
idx = np.where(dist > 0.001) # ORIGINAL 0.01. Seems good at 0.001

# NEW Add plane to point cloud to prevent grasping from under the table
num_table_pts = 100
x_table = np.reshape(np.linspace(np.amin(X[:,0]),np.amax(X[:,0]),num_table_pts),(num_table_pts,1))
y_table = np.reshape(np.linspace(np.amin(X[:,1]),np.amax(X[:,1]),num_table_pts),(num_table_pts,1))
xx, yy = np.meshgrid(x_table, y_table)
z_table = -(a*xx + b*yy + d)/c
# z_table = -(a*x_table + b*y_table + d)/c
# print(x_table.shape)
# print(z_table.shape)
# table_points = np.concatenate((x_table, y_table, z_table), axis=1)
table_points = np.concatenate((np.reshape(xx,(num_table_pts**2,1)), np.reshape(yy,(num_table_pts**2,1)), np.reshape(z_table,(num_table_pts**2,1))), axis=1)
# print(table_points)
# print(np_cloud.shape)
# print(xx.shape)
# print(z_table.shape)
# print(table_points.shape)
np_cloud = np.concatenate([np_cloud,table_points])
# print(np_cloud[-1,:])
# np_cloud = table_points
# print(np_cloud.shape)

# with open("dist_from_plane.csv","w") as test_file: # NEW
#     for i in dist:
#         test_file.write("%.4f, \n" % i)

# print(np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),np.transpose(A)),X[:,2])) # NEW
# print(X.shape) # NEW
print("PLANE COEFFICIENTS:") # NEW
print(a, b, c, d) # NEW

# Publish point cloud and nonplanar indices.
from gpd.msg import CloudIndexed
from std_msgs.msg import Header, Int64
from geometry_msgs.msg import Point

pub = rospy.Publisher('cloud_indexed', CloudIndexed, queue_size=1)

msg = CloudIndexed()
header = Header()
header.frame_id = "/base_link"
header.stamp = rospy.Time.now()
msg.cloud_sources.cloud = point_cloud2.create_cloud_xyz32(header, np_cloud.tolist())
msg.cloud_sources.view_points.append(Point(0,0,0))
for i in xrange(np_cloud.shape[0]):
    msg.cloud_sources.camera_source.append(Int64(0))
for i in idx[0]:
    msg.indices.append(Int64(i))    
s = raw_input('Hit [ENTER] to publish')
pub.publish(msg)
rospy.sleep(2)
print 'Published cloud with', len(msg.indices), 'indices'


# Select a grasp for the robot to execute.
from gpd.msg import GraspConfigList

grasps = [] # global variable to store grasps

def callback(msg):
    print("test")
    global grasps
    grasps = msg.grasps
    
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

# NEW
print grasp
