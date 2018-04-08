import math
import cv2

class Blob:

    def __init__(self, contour):

        self.__contour = contour;

        x, y, w, h = cv2.boundingRect(contour);

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w*h

        self.centerX = (x + x + w) / 2;
        self.centerY = (y + y + h) / 2;

        self.dblDiagonalSize = math.sqrt(math.pow(w, 2) + math.pow(h, 2));

        self.dblAspectRatio = float(w) / float(h);

    def set_contour(self, contour):
        self.__contour = contour

    def get_contour(self):
        return self.__contour

    def __str__(self):
        return ("(x,y): " + str(self.x)+","+str(self.y) + "           (width,height): " + str(self.w) +"," +str(self.h)+
            "               area: " + str(self.w*self.h) + "                   dblAspectRatio:" + str(self.dblAspectRatio) +
            "                         dblDiagonalSize: " + str(self.dblDiagonalSize)
            )

    def isPerson(self):
        print(self)
        if (self.area > 1000 and
                self.dblAspectRatio >= 0.2 and
                self.dblAspectRatio <= 1.2 and
                self.w > 15 and
                self.h> 20 and
                self.dblDiagonalSize > 30.0):
            return True
        return False
