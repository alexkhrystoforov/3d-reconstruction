import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


with np.load('intrinsics-iphoneX-photo-mode.npz') as X:
    K, dist = [X[i] for i in ('name1', 'name2')]


def pointcloud(img1,img2):

    imgL = cv2.imread(img1)
    imgR = cv2.imread(img2)

    # detect sift features for both images
    sift = cv2.xfeatures2d.SIFT_create()
    matcher = Matcher(sift, imgL, imgR)
    good = matcher.BF_matcher()
    print(len(good))
    MIN_MATCH_COUNT = 100

    if len(good) > MIN_MATCH_COUNT:
        pts1 = np.float32([matcher.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([matcher.kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    img_siftmatch = cv2.drawMatches(imgL, matcher.kp1, imgR, matcher.kp2, good, None)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=3.0, mask=None)

    matchesMask = mask.ravel().tolist()

    print("Essential matrix:")
    print(E)

    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    print("Rotation:")
    print(R)

    print("Translation:")
    print(t)

    p1_tmp = np.ones([3, pts1.shape[0]])
    p1_tmp[:2, :] = np.squeeze(pts1).T
    p2_tmp = np.ones([3, pts2.shape[0]])
    p2_tmp[:2, :] = np.squeeze(pts2).T
    # print((np.dot(R, p2_tmp) + t) - p1_tmp)

    # calculate projection matrix for both camera
    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    P_l = np.dot(K, M_l)
    P_r = np.dot(K, M_r)

    # undistort points
    pts1 = pts1[np.asarray(matchesMask) == 1, :, :]
    pts2 = pts2[np.asarray(matchesMask) == 1, :, :]
    pts1_un = cv2.undistortPoints(pts1, K, None)
    pts2_un = cv2.undistortPoints(pts2, K, None)
    pts1_un = np.squeeze(pts1_un)
    pts2_un = np.squeeze(pts2_un)

    point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts1_un.T, pts2_un.T)
    point_3d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_3d[:3, :].T
    print(len(point_3d))
    cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    cv2.imshow('matches', img_siftmatch)
    cv2.waitKey(0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for x, y, z in point_3d:
        ax.scatter(x, y, z, c="r", marker="o")

    plt.show()
    fig.savefig('3D-pointcloud-set-' + str(img1[5]) + '.png')

dir_name = "sets/"

pointcloud(dir_name + "2-L.jpeg", dir_name + "2-R.jpeg")
pointcloud(dir_name + "3-L.jpeg", dir_name + "3-R.jpeg")