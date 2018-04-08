from Point import Point
import cv2
import numpy as np

mousePosition = Point(0,0)

def predictNextPosition(positions):

    predictNextPosition = Point(0,0)
    deltaX = 0.0
    deltaY = 0.0
    samples = 0.0
    lengthPos = len(positions)
    lengthPos2 = lengthPos

    if( lengthPos ==0 ):
        print("Erro: There is no position")
    elif ( lengthPos == 1):
        return positions[0]
    elif (lengthPos == 2):
        deltaX += (positions[1].x - positions[0].x)
        deltaY += (positions[1].y - positions[0].y)

        predictNextPosition.x = positions[-1].x + deltaX
        predictNextPosition.y = positions[-1].y + deltaY

        return predictNextPosition

    elif lengthPos >= 5:
        lengthPos = 5

    iteration = 1
    for p in positions: print p
    #TO FIX
    for i in range(lengthPos,1,-1):
        deltaX += (positions[lengthPos2-iteration].x - positions[lengthPos2-iteration-1].x)*(i-1)
        deltaY += (positions[lengthPos2-iteration].y - positions[lengthPos2-iteration-1].y)*(i-1)
        iteration+=1
        samples += i-1

    deltaX = int(round(deltaX/samples))
    deltaY = int(round(deltaY/samples))

    predictNextPosition.x = positions[-1].x + deltaX
    predictNextPosition.y = positions[-1].y + deltaY

    return predictNextPosition

def onMouse(event, x, y, flags, userData):
    if( event == cv2.EVENT_MOUSEMOVE):
        mousePosition.x = x
        mousePosition.y = y


# mouse callback function
def drawCross(img, center ,color):
    cv2.line(img, (center.x-5, center.y-5), (center.x+5, center.y+5), color,2)
    cv2.line(img, (center.x+5, center.y-5), (center.x-5, center.y+5), color,2)

def main():
    # Create a black image, a window and bind the function to window
    img = np.zeros((800,600,3), np.uint8)
    cv2.namedWindow('predictMousePosition')
    cv2.setMouseCallback('predictMousePosition', onMouse)
    mousePositions = []

    while(True):
        mousePositions.append(mousePosition)

        predictedMousePosition = predictNextPosition(mousePositions)

        print("current position        = " + str(mousePositions[-1].x) + ", " + str(mousePositions[-1].y))
        print("next predicted position = " + str(predictedMousePosition.x) + ", " + str(predictedMousePosition.y))
        print("-----------------------------------------------")


        drawCross(img, mousePositions[-1], (255,0,0))
        #drawCross(img, predictedMousePosition, (0,255,255))

        cv2.imshow('predictMousePosition',img)

        img = np.zeros((512,512,3), np.uint8)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
