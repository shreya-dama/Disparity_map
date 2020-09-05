import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np



def calibrate(side):
    images = [file for file in glob.glob("../../images/task_1/"+side+"_*.png")]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
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
            
    #         Draw and display the corners
    #         img = cv2.drawChessboardCorners(img, (9,6), corners,ret)
    #         cv2.imshow('img',img)
    #         cv2.waitKey(500)
    # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1],None,None)
    # dist = np.array([-0.13615181, 0.53005398, 0, 0, 0]) 

    img = cv2.imread("../../images/task_1/"+side+"_2.png")
    # print(img)
    cv2.imshow('img',img)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    cv2.imshow('newcameramtx',newcameramtx)


    # undistort
    # dst = cv2.undistort(img, mtx, dist, None, mtx)
    # print("dst")

    # # crop the image

    # cv2.imwrite('calibresult.png',dst)

    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,mtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
#     #notice how its almost exactly the same, imagine cv2 is the namespace for cv 
#     #in C++, only difference is FILE_STORGE_WRITE is exposed directly in cv2
#     cv_file = cv2.FileStorage("test.xml", cv2.FILE_STORAGE_WRITE)
#     #creating a random matrix
#     matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     # this corresponds to a key value pair, internally opencv takes your numpy 
#     # object and transforms it into a matrix just like you would do with << 
#     # in c++
#     cv_file.write("my_matrix", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
#     # note you release you don't close() a FileStorage object
#     cv_file.release()
    #print("here")
    path = "../../output/task_1"
    parameter_path = "../../parameters"
    cv2.imwrite(os.path.join(path,side+"_calibresult.png"), dst)
    camera_parameters = cv2.FileStorage(os.path.join(parameter_path,side+"_camera_parameters.xml"), cv2.FILE_STORAGE_WRITE)
    camera_parameters.write(side+"_intrinsic_parameters", mtx)
    camera_parameters.write(side+"_distortion_coefficients", dist)
    camera_parameters.release()


calibrate("left")
calibrate("right")
