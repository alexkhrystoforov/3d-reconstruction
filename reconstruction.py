import cv2
import numpy as np
from matplotlib import pyplot as plt

with np.load('intrinsics-iphoneX-photo-mode.npz') as X:
    K, dist = [X[i] for i in ('name1', 'name2')]

img_path1 = 'sets/3-L.jpeg'
img_path2 = 'sets/3-R.jpeg'

img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)

h, w = img_2.shape[:2]

# Get optimal camera matrix for better undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)
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

# Show disparity map before generating 3D cloud to verify that point cloud will be usable.
plt.imshow(disparity_map, 'gray')
plt.show()

# Generate  point cloud.
print("\nGenerating the 3D map...")

# Get new downsampled width and height
h, w = img_2_undistorted.shape[:2]

# Load focal length.
fx = K[0][0]  # 1025.
# sensor_width = 10
focal_length = fx / w

print(focal_length)

# Perspective transformation matrix
# Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q = np.float32([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1 / focal_length, 0],
                [0, 0, 0, 1]])

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
output_file = 'reconstructed.ply'


# Function to create point cloud file
def create_output(vertices, colors, filename):
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
