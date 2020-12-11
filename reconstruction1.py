import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *

with np.load('intrinsics-iphoneX-photo-mode.npz') as X:
    K, dist = [X[i] for i in ('name1', 'name2')]

imgL_path = 'sets/3-L.jpeg'
imgR_path = 'sets/3-R.jpeg'

imgL = cv2.imread(imgL_path)
imgR = cv2.imread(imgR_path)

h, w = imgR.shape[:2]
sift = cv2.SIFT_create()

matcher = Matcher(sift, imgL, imgR)

# matcher = Matcher(orb, imgLgray, imgRgray)
best_matches = matcher.BF_matcher()

print(len(best_matches))

pts1 = np.float32([matcher.kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts2 = np.float32([matcher.kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=3.0, mask=None)

#print(E)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, dist, K, dist, (w, h), R, t)
print('Q matrix', Q)

# Generate  point cloud.
print("\nGenerating the 3D map...")

# Get new downsampled width and height
# h, w = img_2_undistorted.shape[:2]

# Load focal length.
# fx = K[0][0]  # 1025.
# sensor_width = 10
# focal_length = fx / w
# print(focal_length)

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

img_1_undistorted = cv2.undistort(imgL, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(imgR, K, dist, None, new_camera_matrix)

cv2.namedWindow('1', cv2.WINDOW_NORMAL)
cv2.namedWindow('2', cv2.WINDOW_NORMAL)

cv2.imshow('1', img_1_undistorted)
cv2.imshow('2', img_2_undistorted)
cv2.waitKey(0)

min_disparity = 0
num_disparities = 13 * 16
max_disparity = 13 * 16
window_size = 3

# Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity=min_disparity,
                               numDisparities=num_disparities,
                               blockSize=3,
                               uniquenessRatio=5,
                               speckleWindowSize=5,
                               speckleRange=5,
                               disp12MaxDiff=2,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2)

disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)

# Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
# print(points_3D)
# Get color points
colors = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2RGB)

# Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Define name for output file
output_file = 'reconstructed1.ply'


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors / 255
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


# # Generate point cloud
print("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)

from pyntcloud import PyntCloud

print('read 3d cloud')
human_face = PyntCloud.from_file("reconstructed.ply")
human_face.plot()
