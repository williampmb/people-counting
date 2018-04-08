
import argparse
import datetime
import imutils
import math
import cv2
import numpy as np
import copy
from Blob import Blob
from Point import Point

kernel3x3 = np.ones((3,3), np.uint8)
kernel5x5 = np.ones((5,5), np.uint8)
kernel7x7 = np.ones((7,7), np.uint8)
kernel9x9 = np.ones((9,9), np.uint8)



def matchCurrentFrameBlobsToExistingBlobs(blobs, currentBlobs):
    for existingBlob in blobs:
        existingBlob.isMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()

    for curBlob in currentBlobs:
        indexOfLeastDistance = 0
        leastDistance = 100000.0

        for i in range(0, len(blobs)):
            if blobs[i].isStillBeingTracked:
                distance = Point.distanceBetweenPoints(curBlob.centerPositions[-1], blobs[i].predictedNextPosition)
                if distance < leastDistance:
                    leastDistance = distance
                    indexOfLeastDistance = i

        if leastDistance < curBlob.diagonalSize*1.15 :
            addBlobToExistingBlobs(curBlob, blobs, indexOfLeastDistance)
        else:
            addNewBlob(curBlob, blobs)

    for existingBlob in blobs :
        if not existingBlob.isMatchFoundOrNewBlob:
            existingBlob.numbOfConsecutiveFramesWithoutAMatch +=1
        if existingBlob.numbOfConsecutiveFramesWithoutAMatch >= 5:
            existingBlob.isStillBeingTracked = False

def addBlobToExistingBlobs(curBlob, blobs, index):
    blobs[index].contour = curBlob.contour
    blobs[index].set_bounding_rect(curBlob.contour)

    blobs[index].centerPositions.append(curBlob.centerPositions[-1])

    blobs[index].diagonalSize = curBlob.diagonalSize
    blobs[index].aspectRatio = curBlob.aspectRatio

    blobs[index].isStillBeingTracked = True
    blobs[index].isMatchFoundOrNewBlob = True


def addNewBlob(curBlob, blobs):
    curBlob.isMatchFoundOrNewBlob = True
    blobs.append(curBlob)

def drawAndShowContours(img, contours, nameWindow):

    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow(nameWindow, img)

def drawAndShowBlobs(img, blobs, nameWindow):
    h,w = img.shape[:2]
    imgCurrentBlobs = np.zeros((h,w,3),dtype=np.uint8)
    contours = []

    for b in blobs:
        if(b.isStillBeingTracked):
            contours.append(b.get_contour())

    cv2.drawContours(imgCurrentBlobs, contours, -1, (255,255,255), -1)

    cv2.imshow(nameWindow, imgCurrentBlobs);


def drawBlobInfoOnImage(blobs, img):

    for i in range(0,len(blobs),1):
        if blobs[i].isStillBeingTracked :
            cv2.rectangle(img, (blobs[i].position.x, blobs[i].position.y), (blobs[i].position.x + blobs[i].width, blobs[i].position.y + blobs[i].height), (0,255,255), 2)

            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = blobs[i].diagonalSize/60.0
            fontThickness = int(round(fontScale*1.0))

            posTuple = (blobs[i].position.x, blobs[i].position.y)

            cv2.putText(img, str(i), posTuple, fontFace, fontScale, (0,0,255), fontThickness)


def main():


    video = cv2.VideoCapture("../videos_people/768x576.avi")

    if not video.isOpened():
        print("Video not Opened!")
        return

    #Maybe we should check if the video has at least 2 frames

    _, imgFrame1 = video.read()
    _, imgFrame2 = video.read()

    atFrame = 2
    blobs = []
    #While the video is open and we don't press q key read, process and show a frame
    while(video.isOpened()):

        currentBlobs = []

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

        imgThresh = cv2.dilate(imgThresh, kernel5x5, iterations=2)
        imgThresh = cv2.erode(imgThresh, kernel5x5, iterations=1)

        imgThreshCopy = copy.deepcopy(imgThresh)

        # Contours can be explained simply as a curve joining all the continuous points (along the boundary),
        # having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
        # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
        #im2, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgThreshCopy, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        drawAndShowContours(imgThreshCopy, contours, 'imgContours')

        for x in contours:
            convexHull = cv2.convexHull(x)
            blob = Blob(convexHull)
            if(blob.isPerson()):
                currentBlobs.append(blob)

        drawAndShowBlobs(imgThresh, currentBlobs,"imgCurrentBlobs")

        if atFrame <= 2 :
            for curBlob in currentBlobs:
                blobs.append(curBlob)
        else:
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentBlobs)

        drawAndShowBlobs(imgThresh, blobs, "imgBlobs")

        imgFrame2Copy = copy.deepcopy(imgFrame2)

        drawBlobInfoOnImage(blobs, imgFrame2Copy)

        cv2.imshow('imgFrame2Copy', imgFrame2Copy)

        #cv2.waitKey(0) #for debugging purpose


        # get ready for next iteration

        del currentBlobs[:]

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
