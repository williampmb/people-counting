import math
import cv2
from Point import Point

class Blob:
    id = 0
    conf_area = 0
    conf_min_aspect_ratio = 0
    conf_max_aspect_ratio = 0
    conf_width = 0
    conf_height = 0
    conf_diagonal_size = 0
    conf_contour_area_by_area = 0


    @staticmethod
    def getId():
        Blob.id +=1
        return Blob.id

    def __init__(self, contour):
        self.id = -1
        self.contour = contour
        self.set_bounding_rect(contour)

        self.center = Point(((self.position.x + self.position.x + self.width)/2),
                            ((self.position.y + self.position.y + self.height)/2))

        curCenter = Point(self.center.x, self.center.y)

        self.centerPositions = []
        self.centerPositions.append(curCenter)
        self.predictedNextPosition = self.predictNextPosition()
        self.isStillBeingTracked = True
        self.isMatchFoundOrNewBlob = True

        self.numbOfConsecutiveFramesWithoutAMatch = 0

    def set_bounding_rect(self, contour):
        x, y, w, h = cv2.boundingRect(contour);

        #self.set_bounding_rect(contour)
        self.position = Point(x,y)
        self.width = w
        self.height = h
        self.area = self.width*self.height
        self.diagonalSize = math.sqrt(math.pow(self.width, 2) + math.pow(self.height, 2));
        self.aspectRatio = float(self.width) / float(self.height);

    def set_contour(self, contour):
        self.contour = contour

    def get_contour(self):
        return self.contour

    def __str__(self):
        times =0
        if len(self.centerPositions)>=5:
            times =5
        else:
            times = len(self.centerPositions)-1
        return (" id: " + str(self.id) +
            " (x,y):" + str(self.position.x)+","+str(self.position.y) +
            "    (w,h):" + str(self.width) +"," +str(self.height)+
            "    area:" + str(self.area) +
            "    AS:" + str("%.2f" % self.aspectRatio) +
            "    DS: " + str("%.2f" % self.diagonalSize) +
            #"    area(contour)/area " + str(cv2.contourArea(self.contour))+ "/" + str(self.area) + "= " + str("%.2f" % (cv2.contourArea(self.contour)/float(self.area)))+
            "    area(contour)/area=" + str("%.2f" % (cv2.contourArea(self.contour)/float(self.area)))+
            #"    center: " + ' '.join(str(e.x)+","+str(e.y) for e in self.centerPositions) +
            "    center: " + ' '.join(str(self.centerPositions[-i].x)+","+str(self.centerPositions[-i].y) for i in range(times)) +
            "    tracked: " + str(self.isStillBeingTracked) +
            "    w/tmatch: " + str(self.numbOfConsecutiveFramesWithoutAMatch)
            )

    def predictNextPosition(self):
        numPos = len(self.centerPositions)
        predictedNextPosition = Point(self.centerPositions[-1].x,self.centerPositions[-1].y)

        if ( numPos == 1):
            return predictedNextPosition

        elif (numPos == 2):
            deltaX = (self.centerPositions[1].x - self.centerPositions[0].x)
            deltaY = (self.centerPositions[1].y - self.centerPositions[0].y)

            predictedNextPosition.x = self.centerPositions[-1].x + deltaX
            predictedNextPosition.y = self.centerPositions[-1].y + deltaY

            return predictedNextPosition
        elif numPos == 3:
            sumChangesX = (self.centerPositions[2].x - self.centerPositions[1].x)*2 + (self.centerPositions[1].x - self.centerPositions[0].x)*1
            sumChangesY = (self.centerPositions[2].y - self.centerPositions[1].y)*2 + (self.centerPositions[1].y - self.centerPositions[0].y)*1

            deltaX = int(round(float(sumChangesX)/3.0))
            deltaY = int(round(float(sumChangesY)/3.0))

            predictedNextPosition.x = self.centerPositions[-1].x + deltaX
            predictedNextPosition.y = self.centerPositions[-1].y + deltaY

        elif numPos == 4:

            sumChangesX = ((self.centerPositions[3].x - self.centerPositions[2].x)*3
                        + (self.centerPositions[2].x - self.centerPositions[1].x)*2
                        + (self.centerPositions[1].x - self.centerPositions[0].x)*1 )
            sumChangesY =  ((self.centerPositions[3].y - self.centerPositions[2].y)*3
                        + (self.centerPositions[2].y - self.centerPositions[1].y)*2
                        + (self.centerPositions[1].y - self.centerPositions[0].y)*1)

            deltaX = int(round(float(sumChangesX)/6.0))
            deltaY = int(round(float(sumChangesY)/6.0))

            predictedNextPosition.x = self.centerPositions[-1].x + deltaX
            predictedNextPosition.y = self.centerPositions[-1].y + deltaY

        elif numPos >= 5:

            sumChangesX = ((self.centerPositions[numPos-1].x - self.centerPositions[numPos-2].x)*4
                        + (self.centerPositions[numPos-2].x - self.centerPositions[numPos-3].x)*3
                        + (self.centerPositions[numPos-3].x - self.centerPositions[numPos-4].x)*2
                        + (self.centerPositions[numPos-4].x - self.centerPositions[numPos-5].x)*1)

            sumChangesY = ((self.centerPositions[numPos-1].y - self.centerPositions[numPos-2].y)*4
                        + (self.centerPositions[numPos-2].y - self.centerPositions[numPos-3].y)*3
                        + (self.centerPositions[numPos-3].y - self.centerPositions[numPos-4].y)*2
                        + (self.centerPositions[numPos-4].y - self.centerPositions[numPos-5].y)*1)


            deltaX = int(round(float(sumChangesX)/10.0))
            deltaY = int(round(float(sumChangesY)/10.0))

            predictedNextPosition.x = self.centerPositions[-1].x + deltaX
            predictedNextPosition.y = self.centerPositions[-1].y + deltaY

        self.predictedNextPosition = predictedNextPosition
        return predictedNextPosition

    def isObject(self):
        if (self.area > Blob.conf_area and
                self.aspectRatio >= Blob.conf_min_aspect_ratio and
                self.aspectRatio <= Blob.conf_max_aspect_ratio and
                self.width  > Blob.conf_width and
                self.height > Blob.conf_height and
                self.diagonalSize > Blob.conf_diagonal_size and
                cv2.contourArea(self.contour)/float(self.area) > Blob.conf_contour_area_by_area
                ):
            return True
        return False
