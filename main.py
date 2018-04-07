
import argparse
import datetime
import imutils
import math
import cv2
import numpy as np

def main():

    video = cv2.VideoCapture("../videos_people/768x576.avi")

    if not video.isOpened():
        print("Video not Opened!")

    #While the video is open and we don't press q key read, process and show a frame
    while(video.isOpened() && (cv2.waitKey(1) & 0xFF == ord('q'))):
        # Capture frame-by-frame
        ret, frame = video.read()

        # processing frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display it
        cv2.imshow('frame', gray)

        if cv2.waitKey() & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    
    print("end")

#camera = cv2.VideoCapture("test2.mp4")

if __name__ == "__main__":
    main()
