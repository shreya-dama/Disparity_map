#!/usr/bin/env python2
# -- coding: utf-8 --
"""
Created on Wed Mar  4 16:33:08 2020

@author: shreya
"""
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_3dto2d(side):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = [file for file in sorted(glob.glob("../../images/task_3_and_4/" + side + "_0.png"))]
    object_points = []
    image_points = []
    objp = np.zeros((6 * 9, 3),
                    np.float32)  # 6,9 represents the total number of co-ordinates in y and x direction respectively.
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Iterate through the lists and get the corners of the images
    # plt.figure(figsize = (16,32)) # To plot the (width, height) in inches
    for image in images:  # image will iterate through the list of the paths and consider each path one after another
        img = cv2.imread(image)  # img stores the data of the image located in the path in the form of numpy array and reads the data of pixel in (BGR) format.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray stres the graysacle img
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)  # Prints the 9*6=54 corners (x,y) co-ordinates.
        # if all the corners are present in the gray(image) the image returns True
        if ret == True:
            mod_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(objp)  # objp are added to the object_points
            image_points.append(mod_corners)  # Corners are added to the image_points
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    #         Draw and display the corners
    #         img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
    #         cv2.imshow('img',img)
    #         cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return gray.shape[::-1], object_points, image_points


image_size, object_points, left_image_points = extract_3dto2d("left")
_, _, right_image_points = extract_3dto2d("right")

#stereo_parameters = cv2.FileStorage("../../stereo_parameters.xml", cv2.FILE_STORAGE_READ)
stereo_parameters = cv2.FileStorage("../../parameters/stereo_parameters.xml", cv2.FILE_STORAGE_READ)
left_intrinsic_parameters = stereo_parameters.getNode("left_intrinsic_parameters").mat()
left_distortion_coefficients = stereo_parameters.getNode("left_distortion_coefficients").mat()
right_intrinsic_parameters = stereo_parameters.getNode("right_intrinsic_parameters").mat()
right_distortion_coefficients = stereo_parameters.getNode("right_distortion_coefficients").mat()
R = stereo_parameters.getNode("R").mat()
T = stereo_parameters.getNode("T").mat()
E = stereo_parameters.getNode("E").mat()
F = stereo_parameters.getNode("F").mat()
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_intrinsic_parameters,
                                                  left_distortion_coefficients, right_intrinsic_parameters,
                                                  right_distortion_coefficients, image_size, R, T)
map_left_1, map_left_2 = cv2.initUndistortRectifyMap(left_intrinsic_parameters, left_distortion_coefficients, R1, P1, image_size, cv2.CV_8UC1)
map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
    right_intrinsic_parameters, right_distortion_coefficients, R2, P2, image_size, cv2.CV_8UC1)

distort_left_images_4 = cv2.imread("../../images/task_3_and_4/left_6.png")
distort_right_images_4 = cv2.imread("../../images/task_3_and_4/right_6.png")
#cv2.imwrite('distort_left_images_4.png', distort_left_images_4)
#cv2.imwrite('distort_right_images_4.png', distort_right_images_4)

undistort_left_images_4 = cv2.undistort(distort_left_images_4, left_intrinsic_parameters,
                                        left_distortion_coefficients)
undistort_right_images_4 = cv2.undistort(distort_right_images_4, right_intrinsic_parameters,
                                         right_distortion_coefficients)
#cv2.imwrite('undistort_left_images_4.png', undistort_left_images_4)
#cv2.imwrite('undistort_right_images_4.png', undistort_right_images_4)

rectified_left_images_4 = cv2.remap(distort_left_images_4, map_left_1, map_left_2, cv2.INTER_LINEAR)
rectified_right_images_4 = cv2.remap(distort_right_images_4, map_right_1, map_right_2, cv2.INTER_LINEAR)

#cv2.imwrite('rectified_left_images_4.png', rectified_left_images_4)
#cv2.imwrite('rectified_right_images_4.png', rectified_right_images_4)

left_matcher = cv2.StereoSGBM_create(minDisparity=0,
                                     numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                                     blockSize=5,
                                     P1=8 * 3 * 3 ** 2,
                                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                                     P2=32 * 3 * 3 ** 2,
                                     disp12MaxDiff=1,
                                     uniquenessRatio=15,
                                     speckleWindowSize=0,
                                     speckleRange=2,
                                     preFilterCap=63,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                     )

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 10000
sigma = 1.2
visual_multiplier = 0.5

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

displ = left_matcher.compute(rectified_left_images_4, rectified_right_images_4)  # .astype(np.float32)/16
dispr = right_matcher.compute(rectified_right_images_4, rectified_left_images_4)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, rectified_left_images_4, None, dispr)  # important to put "imgL" here!!!

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
#disparity_map = cv2.imwrite('Disparity Map.png', filteredImg)

depth = cv2.reprojectImageTo3D(displ, Q)
print(depth)

path = "../../output/task_4"
cv2.imwrite(os.path.join(path,"Rectified Image Left.png"), rectified_left_images_4)
cv2.imwrite(os.path.join(path,"Rectified Image Right.png"), rectified_right_images_4)
cv2.imwrite(os.path.join(path,"Disparity.png"), filteredImg)


#im_h_resize = hconcat_resize_min([rectified_left_images_4, rectified_right_images_4, filteredImg])
cv2.imshow("Rectified Image Left",rectified_left_images_4)
# cv2.moveWindow(rectified_left_images_4,0,0)
cv2.imshow("Rectified Image Right",rectified_right_images_4)
# cv2.moveWindow(rectified_right_images_4,0,1)
cv2.imshow("Disparity",filteredImg)
# cv2.moveWindow(filteredImg,0,2)
# im_v = cv2.vconcat([rectified_left_images_4, rectified_right_images_4,filteredImg])
# numpy_horizontal_concat = np.concatenate((rectified_left_images_4, rectified_right_images_4,filteredImg), axis=2)
# final_concat=np.concatenate((numpy_horizontal_concat, filteredImg), axis=1)
#cv2.imshow("Rectified Image Right", im_h_resize)
#path = "../../output/task_1"
#cv2.imwrite(os.path.join(path,side+"_calibresult.png"), dst)
cv2.waitKey(0)