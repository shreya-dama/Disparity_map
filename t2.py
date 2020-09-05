import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d



def extract_3dto2d(side):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = [file for file in sorted(glob.glob("../../images/task_2/"+side+"_0.png"))]
    object_points=[]
    image_points=[]
    objp = np.zeros((6*9,3), np.float32) # 6,9 represents the total number of co-ordinates in y and x direction respectively.
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Iterate through the lists and get the corners of the images
    #plt.figure(figsize = (16,32)) # To plot the (width, height) in inches
    for image in images: # image will iterate through the list of the paths and consider each path one after another 
        img = cv2.imread(image) # img stores the data of the image located in the path in the form of numpy array and reads the data of pixel in (BGR) format.    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray stres the graysacle img
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None) # Prints the 9*6=54 corners (x,y) co-ordinates.
        # if all the corners are present in the gray(image) the image returns True
        if ret == True :
            mod_corners = cv2.cornerSubPix(gray,corners,(11, 11),(-1,-1),criteria)
            object_points.append(objp) #objp are added to the object_points
            image_points.append(mod_corners) # Corners are added to the image_points 
            img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
    #         Draw and display the corners
    #         img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
    #         cv2.imshow('img',img)
    #         cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return gray.shape[::-1], object_points, image_points

def stereo_calibrate():

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    left_camera_parameters = cv2.FileStorage("../../parameters/left_camera_parameters.xml", cv2.FILE_STORAGE_READ)
    temp_left_intrinsic_parameters = left_camera_parameters.getNode("left_intrinsic_parameters").mat()
    temp_left_distortion_coefficients = left_camera_parameters.getNode("left_distortion_coefficients").mat()
    left_camera_parameters.release()
    
    right_camera_parameters = cv2.FileStorage("../../parameters/right_camera_parameters.xml", cv2.FILE_STORAGE_READ)
    temp_right_intrinsic_parameters = right_camera_parameters.getNode("right_intrinsic_parameters").mat()
    temp_right_distortion_coefficients = right_camera_parameters.getNode("right_distortion_coefficients").mat()
    right_camera_parameters.release()
    
    
    image_size, object_points, left_image_points = extract_3dto2d("left")
    _, _, right_image_points = extract_3dto2d("right")
    
    ret, left_intrinsic_parameters, left_distortion_coefficients, right_intrinsic_parameters, right_distortion_coefficients, R, T, E, F = cv2.stereoCalibrate(
        object_points, left_image_points, right_image_points, temp_left_intrinsic_parameters, 
        temp_left_distortion_coefficients, temp_right_intrinsic_parameters, 
        temp_right_distortion_coefficients, image_size, criteria=stereocalib_criteria)

    parameter_path = "../../parameters"
    # os.path.join(parameter_path,"stereo_parameters.xml")
    
    stereo_parameters = cv2.FileStorage(os.path.join(parameter_path,"stereo_parameters.xml"), cv2.FILE_STORAGE_WRITE)
    stereo_parameters.write("left_intrinsic_parameters", left_intrinsic_parameters)
    stereo_parameters.write("left_distortion_coefficients", left_distortion_coefficients)
    stereo_parameters.write("right_intrinsic_parameters", right_intrinsic_parameters)
    stereo_parameters.write("right_distortion_coefficients", right_distortion_coefficients)
    stereo_parameters.write("R", R)
    stereo_parameters.write("T", T)
    stereo_parameters.write("E", E)
    stereo_parameters.write("F", F)
    stereo_parameters.release()
    
#     print(x)
#     print(left_intrinsic_parameters)
#     print(d1)
#     print(left_distortion_coefficients)
#     print(mat_rat2)
#     print(right_intrinsic_parameters)
#     print(d2)
#     print(right_distortion_coefficients)
    
    undistorted_left_image_points = cv2.undistortPoints(np.reshape(left_image_points, (54, 1, 2)),
                                                  left_intrinsic_parameters, left_distortion_coefficients)
    undistorted_right_image_points = cv2.undistortPoints(np.reshape(right_image_points, (54, 1, 2)),
                                                   right_intrinsic_parameters, right_distortion_coefficients)
    
    
    normalized_undistorted_left = np.transpose(np.reshape((undistorted_left_image_points), (54, 2)))
    normalized_undistorted_right = np.transpose(np.reshape((undistorted_right_image_points), (54, 2)))
    
    
    P_0 = np.concatenate((np.identity(3),  np.zeros((3, 1))), axis=1)
    P_1 = np.concatenate((R, T), axis=1)
    
    points_4d = cv2.triangulatePoints(P_0, P_1, normalized_undistorted_left, normalized_undistorted_right)
    
    points_3d=np.divide(points_4d[:3],points_4d[3])
    
    print(points_3d)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x_line = points_3d[0]
    y_line = points_3d[1]
    z_line = points_3d[2]
    #ax.plot3D(x_line, y_line, z_line, 'gray')
    ax.scatter(x_line, y_line, z_line, 'gray')
    a = plt.show()
    path = "../../output/task_2"
    fig.savefig(os.path.join(path,'plot.png'))
    #cv2.imwrite('plot.png', a)
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_intrinsic_parameters, 
                                                      left_distortion_coefficients, right_intrinsic_parameters, 
                                                      right_distortion_coefficients, image_size, R, T)

    parameter_path = "../../parameters"
    # os.path.join(parameter_path,"stereo_rectified_parameters.xml")

    stereo_rectified_parameters = cv2.FileStorage(os.path.join(parameter_path,"stereo_rectified_parameters.xml"), cv2.FILE_STORAGE_WRITE)
    stereo_rectified_parameters.write("R1", R1)
    stereo_rectified_parameters.write("R2", R2)
    stereo_rectified_parameters.write("P1", P1)
    stereo_rectified_parameters.write("P2", P2)
    stereo_rectified_parameters.write("Q", Q)
    stereo_rectified_parameters.write("roi1", roi1)
    stereo_rectified_parameters.write("roi2", roi2)
    stereo_rectified_parameters.release()
#     print(P1)
#     print(left_intrinsic_parameters)

    map_left_1, map_left_2  = cv2.initUndistortRectifyMap(
        left_intrinsic_parameters,left_distortion_coefficients, R1, P1, image_size, cv2.CV_32FC1)
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(
        right_intrinsic_parameters, right_distortion_coefficients, R2, P2, image_size, cv2.CV_32FC1)
    
    distort_left_images_0 = cv2.imread("../../images/task_2/left_0.png")
    distort_left_images_1 = cv2.imread("../../images/task_2/left_1.png")

    #os.path.join(path,'rectified_left_images_1.png')
    cv2.imwrite(os.path.join(path,'distort_left_images_0.png'), distort_left_images_0)
    cv2.imwrite(os.path.join(path,'distort_left_images_1.png'), distort_left_images_1)
    
    undistort_left_images_0 = cv2.undistort(distort_left_images_0, left_intrinsic_parameters,
                                            left_distortion_coefficients)
    undistort_left_images_1 = cv2.undistort(distort_left_images_1, right_intrinsic_parameters,
                                            right_distortion_coefficients)
    cv2.imwrite(os.path.join(path,'undistort_left_images_0.png'), undistort_left_images_0)
    cv2.imwrite(os.path.join(path,'undistort_left_images_1.png'), undistort_left_images_1)

    rectified_left_images_0 = cv2.remap(distort_left_images_0, map_left_1, map_left_2, cv2.INTER_LINEAR)
    rectified_left_images_1 = cv2.remap(distort_left_images_1, map_right_1, map_right_2, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(path,'rectified_left_images_0.png'), rectified_left_images_0)
    cv2.imwrite(os.path.join(path,'rectified_left_images_1.png'), rectified_left_images_1)
    

    
stereo_calibrate()
