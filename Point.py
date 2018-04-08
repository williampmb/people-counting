import math

class Point:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __str__(self):
        return (str(self.x)+ " " + str(self.y))

    @staticmethod
    def distanceBetweenPoints(p1, p2):
        x = abs(p1.x - p2.x)
        y = abs(p1.y - p2.y)

        distance = float(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))

        return distance
