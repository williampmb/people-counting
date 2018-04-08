
import argparse
import datetime
import imutils
import math
import cv2
import numpy as np
import copy
from Blob import Blob



def main():

    atFrame = 1
    video = cv2.VideoCapture("../videos_people/768x576.avi")

    if not video.isOpened():
        print("Video not Opened!")

    #Maybe we should check if the video has at least 2 frames

    _, imgFrame1 = video.read()
    _, imgFrame2 = video.read()

    #While the video is open and we don't press q key read, process and show a frame
    while(video.isOpened()):

        imgFrame1Copy = copy.deepcopy(imgFrame1)
        imgFrame2Copy = copy.deepcopy(imgFrame2)

        imgFrame1Copy = cv2.cvtColor(imgFrame1Copy, cv2.COLOR_BGR2GRAY)
        imgFrame2Copy = cv2.cvtColor(imgFrame2Copy, cv2.COLOR_BGR2GRAY)

        imgFrame1Copy = cv2.GaussianBlur(imgFrame1Copy, (5,5), 0)
        imgFrame2Copy = cv2.GaussianBlur(imgFrame2Copy, (5,5), 0)

        imgDifference = cv2.absdiff(imgFrame1Copy, imgFrame2Copy)

        # ret value is used for Otsu's Binarization if we want to
        # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
        ret, imgThresh = cv2.threshold(imgDifference, 30, 255.0, cv2.THRESH_BINARY)

        cv2.imshow('imgThresh', imgThresh)

        kernel3x3 = np.ones((3,3), np.uint8)
        kernel5x5 = np.ones((5,5), np.uint8)
        kernel7x7 = np.ones((7,7), np.uint8)
        kernel9x9 = np.ones((9,9), np.uint8)

        imgThresh = cv2.dilate(imgThresh, kernel5x5, iterations=2)
        imgThresh = cv2.erode(imgThresh, kernel5x5, iterations=1)

        imgThreshCopy = copy.deepcopy(imgThresh)

        # Contours can be explained simply as a curve joining all the continuous points (along the boundary),
        # having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
        # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
        #im2, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgThreshCopy, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(imgThreshCopy, contours, -1, (0,255,0), 3)

        cv2.imshow('imgContours', imgThreshCopy)

        convexHulls = []
        blobs = []

        h,w = imgThresh.shape[:2]
        imgConvexHull = np.zeros((h,w,3),dtype=np.uint8)

        for x in contours:
            hull = cv2.convexHull(x)
            convexHulls.append(hull)

        for convexHull in convexHulls:
            blob = Blob(convexHull)

            if(blob.isPerson()):
                blobs.append(blob)

        del convexHulls[:]

        for blob in blobs:
            convexHulls.append(blob.get_contour())

        cv2.drawContours(imgConvexHull, convexHulls, -1, (255,255,255), -1)

        cv2.imshow("imgConvexHulls", imgConvexHull);

        imgFrame2Copy = copy.deepcopy(imgFrame2)

        for blob in blobs:
            cv2.rectangle(imgFrame2Copy, (blob.x, blob.y), (blob.x + blob.w, blob.y + blob.h), (0,255,255), 2)
            cv2.circle(imgFrame2Copy, (blob.centerX, blob.centerY), 3, (255,255,0), -1)

        cv2.imshow('imgFrame2Copy', imgFrame2Copy)

        # get ready for next iteration

        imgFrame1 = copy.deepcopy(imgFrame2)

        if( (video.get(cv2.CAP_PROP_POS_FRAMES) +1 ) < ( video.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, imgFrame2 = video.read()
        else:
            print("end of video")
            break;

        atFrame +=1
        #print("frame: " +  str(count))
        if cv2.waitKey() & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    print("end")


if __name__ == "__main__":
    main()
