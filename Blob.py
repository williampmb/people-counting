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

        self.centerX = (x + x + w) / 2;
        self.centerY = (y + y + h) / 2;

        self.dblDiagonalSize = math.sqrt(math.pow(w, 2) + math.pow(h, 2));

        self.dblAspectRatio = w / h;

    def set_contour(self, contour):
        self.__contour = contour

    def get_contour(self):
        return self.__contour

    def __str__(self):
        return "x: " + str(self.x)

    def isPerson(self):
        return True
