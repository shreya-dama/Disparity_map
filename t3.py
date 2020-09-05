import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d


def euclidean_distance(point_1, point_2):
    return math.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2)


def keypoints(kp, d):
    temp = []
    for i in range(len(kp) - 2):
        for j in range(i + 1, len(kp) - 2):
            if euclidean_distance(kp[i].pt, kp[j].pt) < d:
                if kp[i].response <= kp[j].response:
                    temp.append(i)
                else:
                    temp.append(j)
                    j = len(kp)
    temp = list(dict.fromkeys(temp))
    temp = sorted(temp, reverse=True)
    for i in temp:
        kp.pop(i)
    return kp


def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def ORB(dst):
    orb = cv2.ORB_create()
    kp = orb.detect(dst, None)
    kp = keypoints(kp, 8)
    kp, des = orb.compute(dst, kp)
    return kp, des


def load(side):
    images = [file for file in glob.glob("../../images/task_3_and_4/" + side + "_0.png")]
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


left_image = load("left")
right_image = load("right")

stereo_parameters = cv2.FileStorage("../../parameters/stereo_parameters.xml", cv2.FILE_STORAGE_READ)
left_intrinsic_parameters = stereo_parameters.getNode("left_intrinsic_parameters").mat()
left_distortion_coefficients = stereo_parameters.getNode("left_distortion_coefficients").mat()
right_intrinsic_parameters = stereo_parameters.getNode("right_intrinsic_parameters").mat()
right_distortion_coefficients = stereo_parameters.getNode("right_distortion_coefficients").mat()
R = stereo_parameters.getNode("R").mat()
T = stereo_parameters.getNode("T").mat()
E = stereo_parameters.getNode("E").mat()
F = stereo_parameters.getNode("F").mat()
stereo_parameters.release()

img1 = undistort(left_image, left_intrinsic_parameters, left_distortion_coefficients)
img2 = undistort(right_image, right_intrinsic_parameters, right_distortion_coefficients)

kp1, des1 = ORB(img1)
kp2, des2 = ORB(img2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:15], None, matchColor=(0, 255, 0), flags=2)

fig = plt.figure()
plt.imshow(img3)
plt.show()
path = "../../output/task_3"
fig.savefig(os.path.join(path,'plot1.png'))

kp1_coord = []
kp2_coord = []

for mat in matches:
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt
    kp1_coord.append((x1, y1))
    kp2_coord.append((x2, y2))

kp1_cord = np.array(kp1_coord, dtype='float32')
kp2_cord = np.array(kp2_coord, dtype='float32')
temp_kp2_cord = np.array([], dtype='float32')
temp_kp1_cord = np.array([], dtype='float32')
undistort_kp1_cord = cv2.undistortPoints(np.reshape(kp1_cord, (len(kp1_coord), 1, 2)), left_intrinsic_parameters,
                                         left_distortion_coefficients)
undistort_kp2_cord = cv2.undistortPoints(np.reshape(kp2_cord, (len(kp2_coord), 1, 2)), right_intrinsic_parameters,
                                         right_distortion_coefficients)

temp_kp1_cord = cv2.convertPointsToHomogeneous(undistort_kp1_cord)
temp_kp2_cord = cv2.convertPointsToHomogeneous(undistort_kp1_cord)

homo_kp1_cord = temp_kp1_cord[:, -1].T
homo_kp2_cord = temp_kp2_cord[:, -1].T
mod_kp1_cord = np.delete(homo_kp1_cord, 2, 0)
mod_kp2_cord = np.delete(homo_kp2_cord, 2, 0)

P_0 = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
P_1 = np.concatenate((R, T), axis=1)

points_4d = cv2.triangulatePoints(P_0, P_1, mod_kp1_cord, mod_kp2_cord)
points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
points_3d = points_3d[:, -1].T

fig = plt.figure()
ax = plt.axes(projection="3d")
x_line = points_3d[0]
y_line = points_3d[1]
z_line = points_3d[2]
ax.scatter(x_line, y_line, z_line, 'gray')
plt.show()
fig.savefig(os.path.join(path,'plot2.png'))