
import argparse
import datetime
import imutils
import math
import cv2
import numpy as np
import copy
import yaml

from Blob import Blob
from Point import Point

#load from config.yaml based on id
video_id = 3

#load from config.yaml
debug_mode = None
debugGaussian = None
debugThreshold = None
debug_dilate = None
debug_erode = None
debug_crossed_blobs = None
debug_all_current_blobs = None

case = 0
videopath = ""
line = []
max_track_frames = 15

gaussiankernel = 2
threshold_value = 0
kernel_dilate1 = None
kernel_erode1 = None
kernel_dilate2 = None
kernel_erode2 = None

def getKernel(numb):
    if numb == 3:
        return np.ones((3,3), np.uint8)
    if numb == 5:
        return np.ones((5,5), np.uint8)
    if numb == 7:
        return np.ones((7,7), np.uint8)
    if numb == 9:
        return np.ones((9,9), np.uint8)
    if numb == 15:
        return np.ones((15,15), np.uint8)

def getTuple(numb):
    if numb == 5:
        return (5,5)
    if numb == 3:
        return (3,3)
    if numb == 15:
        return (15,15)
    if numb == 9:
        return (9,9)
    if numb == 7:
        return (7,7)

def yaml_loader(filepath):
    # Loads a yaml file
    with open(filepath,"r") as file_descriptor:
        data = yaml.load(file_descriptor)
    return data

def load_config(filepath):
    #filepath = "config.yaml"
    data = yaml_loader(filepath)

    #load config
    config = data.get('config')
    global debug_mode, debugGaussian, debug_all_current_blobs
    global debugThreshold, debug_dilate, debug_erode, debug_crossed_blobs
    debug_mode = config.get('debug_mode')
    debugGaussian = config.get('debugGaussian')
    debugThreshold = config.get('debugThreshold')
    debug_erode = config.get('debug_erode')
    debug_dilate = config.get('debug_dilate')
    debug_crossed_blobs = config.get('debug_crossed_blobs')
    debug_all_current_blobs = config.get('debug_all_current_blobs')

    videos = data.get('videos')
    #load video and blobs features
    global video_id
    video_name = "video_"+str(video_id)
    video_config = videos.get(video_name)

    global videopath
    videopath = video_config.get('filepath')

    global case, max_track_frames, line
    max_track_frames = video_config.get('max_track_frames')
    case = video_config.get('case')
    point1Confg = video_config.get('linePoint1')
    point2Confg = video_config.get('linePoint2')
    point1 = Point(point1Confg.get('x'),point1Confg.get('y'))
    point2 = Point(point2Confg.get('x'),point2Confg.get('y'))
    line = [point1,point2]

    blob_config = video_config.get('blob')
    Blob.conf_area = blob_config.get('area')
    Blob.conf_min_aspect_ratio = blob_config.get('min_aspect_ratio')
    Blob.conf_max_aspect_ratio = blob_config.get('max_aspect_ratio')
    Blob.conf_width = blob_config.get('width')
    Blob.conf_height = blob_config.get('height')
    Blob.conf_diagonal_size = blob_config.get('diagonal_size')
    Blob.conf_contour_area_by_area = blob_config.get('contour_area_by_area')

    global gaussian_kernel, threshold_value, kernel_dilate1, kernel_erode1, kernel_dilate2, kernel_erode2
    gaussian_kernel = getTuple(video_config.get('gaussian_kernel'))
    threshold_value = video_config.get('threshold_value')

    kernel_dilate1 = getKernel(video_config.get('kernel_dilate1'))
    kernel_erode1 = getKernel(video_config.get('kernel_erode1'))
    kernel_dilate2 = getKernel(video_config.get('kernel_dilate2'))
    kernel_erode2 =  getKernel(video_config.get('kernel_erode2'))

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

        #best result with 0.7
        if leastDistance < curBlob.diagonalSize*0.80 :
            addBlobToExistingBlobs(curBlob, blobs, indexOfLeastDistance)
        else:
            addNewBlob(curBlob, blobs)

    for existingBlob in blobs :
        if not existingBlob.isMatchFoundOrNewBlob:
            existingBlob.numbOfConsecutiveFramesWithoutAMatch +=1
        if existingBlob.numbOfConsecutiveFramesWithoutAMatch >= max_track_frames:
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
    curBlob.id = Blob.getId()
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

            cv2.putText(img, str(blobs[i].id), posTuple, fontFace, fontScale, (0,0,255), fontThickness)

def checkIfBlobsCossedTheLine(blobs, line, peopleCount, seenPeople, case):
    atLeastOneBlobCrossedTheLine = False

    for b in blobs:
        if b.isStillBeingTracked and len(b.centerPositions) >=4 :
            if(case == 1):
                prevFrameIndex = len(b.centerPositions) - 3
                curFrameIndex = len(b.centerPositions) - 1
                blobMinY = b.centerPositions[prevFrameIndex].y
                blobMaxY = b.centerPositions[curFrameIndex].y
                blobMaxX = b.centerPositions[curFrameIndex].x
                if(blobMinY < line[0].y and blobMaxY >= line[1].y
                    and blobMaxX <= line[1].x):
                    if not b.id in seenPeople:
                        seenPeople.add(b.id)
                        if debug_crossed_blobs:
                            print(str(b))
                            print(str(b.centerPositions[prevFrameIndex].y) + ">"+ str(line[1].y) + ">=" + str(b.centerPositions[curFrameIndex].y))
                            print (str(b.id) + " have crossed the line")
                        peopleCount[0] += 1
                        atLeastOneBlobCrossedTheLine = True

    return atLeastOneBlobCrossedTheLine

def drawPeopleCounterOnImage(peopleCount, img, width, height):
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    #fontScale = float(width*height/450000.0)
    fontScale = float(width*height/450000.0)*5
    fontThickness = round(fontScale*0.9)

    textSize,_ = cv2.getTextSize(str(peopleCount[0]), int(fontFace), fontScale, int(fontThickness))

    w = textSize[0]
    h = textSize[1]
    textBottonLeftPositionX = int(width -1  - int(float(w*1.25)))

    textBottonLeftPositionY = int(float(h*1.25))

    cv2.putText(img,str(peopleCount[0]), (textBottonLeftPositionX,textBottonLeftPositionY), fontFace, fontScale,(0,0,255), int(fontThickness))


def main():
    filepath = videopath
    print filepath
    video = cv2.VideoCapture("./"+videopath)

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float video.get(3)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float video.get(4)

    if not video.isOpened():
        print("Video not Opened!")
        return

    #Maybe we should check if the video has at least 2 frames

    #Read the first and second frame of the video to start doing processing on them
    _, imgFrame1 = video.read()
    _, imgFrame2 = video.read()

    #start framecount as 2 because we just read 2 frames
    atFrame = 2
    #up to this point we have none blobs yet
    blobs = []

    # To count people and pass it as a parameter. It doens't work with primitive variables (int)
    peopleCount = [0]
    seenPeople = set()

    #While the video is open and we don't press q key read, process and show a frame
    while(video.isOpened()):

        #for every frame, check how many blobs are in the screen
        currentBlobs = []

        imgFrame1Copy = copy.deepcopy(imgFrame1)
        imgFrame2Copy = copy.deepcopy(imgFrame2)

        imgFrame1Copy = cv2.cvtColor(imgFrame1Copy, cv2.COLOR_BGR2GRAY)
        imgFrame2Copy = cv2.cvtColor(imgFrame2Copy, cv2.COLOR_BGR2GRAY)


        if(debugGaussian and debug_mode):
            cv2.imshow('gaussianBlurBefore-Img1', imgFrame1Copy)
            cv2.imshow('gaussianBlurBefore-Img2', imgFrame2Copy)

        imgFrame1Copy = cv2.GaussianBlur(imgFrame1Copy, gaussian_kernel, 0)
        imgFrame2Copy = cv2.GaussianBlur(imgFrame2Copy, gaussian_kernel, 0)

        if(debugGaussian and debug_mode):
            cv2.imshow('gaussianBlurAfter-Img1', imgFrame1Copy)
            cv2.imshow('gaussianBlurAfter-Img2', imgFrame2Copy)

        imgDifference = cv2.absdiff(imgFrame1Copy, imgFrame2Copy)

        if(debugGaussian and debug_mode):
            cv2.imshow('dif-Img1-Img2', imgDifference)
        # ret value is used for Otsu's Binarization if we want to
        # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
        ret, imgThresh = cv2.threshold(imgDifference, threshold_value, 255.0, cv2.THRESH_BINARY)

        if debugThreshold and debug_mode:
            cv2.imshow('imgThresh', imgThresh)

        #all the pixels near boundary will be discarded depending upon the size of kernel. erosion removes white noises
        imgThresh = cv2.dilate(imgThresh, kernel_dilate1, iterations=1)
        if debug_dilate:
            cv2.imshow('dilate-dilate1', imgThresh)
        imgThresh = cv2.erode(imgThresh, kernel_erode1, iterations=1)
        if debug_erode:
            cv2.imshow('dilate-erode1', imgThresh)

        imgThresh = cv2.dilate(imgThresh, kernel_dilate2, iterations=1)
        if debug_dilate:
            cv2.imshow('dilate-dilate2', imgThresh)
        imgThresh = cv2.erode(imgThresh, kernel_erode2, iterations=1)
        if debug_erode:
            cv2.imshow('dilate-erode2', imgThresh)

        imgThreshCopy = copy.deepcopy(imgThresh)

        # Contours can be explained simply as a curve joining all the continuous points (along the boundary),
        # having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
        # https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
        #im2, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgThreshCopy, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        drawAndShowContours(imgThreshCopy, contours, 'imgContours')

        #up here we made all processing image stuff and now we need to work with the info we extrated from the image

        #for every thing it's identified on the screen, check if it is a people
        for x in contours:
            convexHull = cv2.convexHull(x)
            blob = Blob(convexHull)
            if(blob.isObject()):
                currentBlobs.append(blob)

        drawAndShowBlobs(imgThresh, currentBlobs, "imgCurrentBlobs")

        if atFrame <= 2 :
            #if it is first iteration there is no comparison, add curBlos to blobs
            for curBlob in currentBlobs:
                curBlob.id = Blob.getId()
                blobs.append(curBlob)
        else:
            #otherwise check if the curblob is releated to a previous blob and match them
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentBlobs)

        if debug_all_current_blobs:
            for b in blobs:
                print b

        drawAndShowBlobs(imgThresh, blobs, "imgBlobs")

        imgFrame2Copy = copy.deepcopy(imgFrame2)

        drawBlobInfoOnImage(blobs, imgFrame2Copy)

        #check if the blob crossed the explained
        atLeastOneBlobCrossedTheLine = checkIfBlobsCossedTheLine(blobs, line, peopleCount, seenPeople, case)

        #if it has cross draw a colorful line
        if atLeastOneBlobCrossedTheLine:
            cv2.line(imgFrame2Copy, (line[0].x,line[0].y), (line[1].x,line[1].y), (255, 0, 255), 2) #yellow line
        else:
            cv2.line(imgFrame2Copy, (line[0].x,line[0].y), (line[1].x,line[1].y), (0, 255, 255), 2)

        #draw the counter
        drawPeopleCounterOnImage(peopleCount, imgFrame2Copy, width, height)

        cv2.imshow('imgFrame2Copy', imgFrame2Copy)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if debug_mode and cv2.waitKey() & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    print("end")


if __name__ == "__main__":
    config_path = "config.yaml"
    load_config(config_path)
    main()
