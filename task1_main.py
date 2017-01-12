import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

# ******* WRITE YOUR FUNCTION, VARIABLES etc HERE

lower_orange=np.array([32,192,220])
upper_orange=np.array([90,230,255])

def dig(x,y):
    return ((x//100)+(3*(y//100))+1)
    # if x<100 and y<100:
    #     return 1
    # elif x>100 and x<200 and y<100:
    #     return 2
    # elif x>200 and x<300 and y<100:
    #     return 3
    # elif x<100 and y>100 and y<200:
    #     return 4
    # elif x>100 and x<200 and y>100 and y<200:
    #     return 5
    # elif x>200 and x<300 and y>100 and y<200:
    #     return 6
    # elif x<100 and y>200 and y<300:
    #     return 7
    # elif x>100 and x<200 and y>200 and y<300:
    #     return 8
    # elif x>200 and x<300 and y>200 and y<300:
    #     return 9

def digc(x,y):
    return ((x//100)+(4*(y//100))+1)

def colorfun(x,y,img):
    #plt.imshow(img)
    #plt.show()
    lower_green=np.array([0,230,0])
    upper_green=np.array([120,255,120])
    green=[lower_green,upper_green,'green']

    lower_red=np.array([0,0,230])
    upper_red=np.array([120,120,255])
    red=[lower_red,upper_red,'red']

    lower_yellow=np.array([0,230,230])
    upper_yellow=np.array([120,255,255])
    yellow=[lower_yellow,upper_yellow,'yellow']

    lower_blue=np.array([230,0,0])
    upper_blue=np.array([255,120,120])
    blue=[lower_blue,upper_blue,'blue']

    colors=[green,red,yellow,blue]

    imag=img[y-1:y+1,x-1:x+1]
    #plt.imshow(imag)
    #plt.show()
    tempnum=0
    for color in colors:
        mask=cv2.inRange(imag,color[0],color[1])
        #cv2.imshow('mask',mask)
        if(mask[1][1]!=0):
            break
        tempnum=tempnum+1
    if(tempnum==0):
        return "green"
    elif(tempnum==1):
        return "red"
    elif(tempnum==2):
        return "yellow"
    elif(tempnum==3):
        return "blue"

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self,c):
        shape = "unidentified"
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            (x,y,w,h) = cv2.boundingRect(approx)
            ar = w/float(h)
            shape = "4-sided"
        else:
            shape = "Circle"
        return shape


def main(board_filepath, container_filepath):
    bard = cv2.imread(board_filepath,cv2.IMREAD_COLOR)
    board=cv2.resize(bard,(300,300))
    mask=cv2.inRange(board,lower_orange,upper_orange)
    mask=cv2.bitwise_not(mask)
    board=cv2.bitwise_and(board,board,mask=mask)

    cant = cv2.imread(container_filepath,cv2.IMREAD_COLOR)
    container = cv2.resize(cant,(400,400))
    maskc=cv2.inRange(container,lower_orange,upper_orange)
    maskc=cv2.bitwise_not(maskc)
    container = cv2.bitwise_and(container,container,mask=maskc)

    #filter
    kernel=np.ones((3,3),np.uint8)
    board=cv2.morphologyEx(board,cv2.MORPH_OPEN,kernel)
    container=cv2.morphologyEx(container,cv2.MORPH_OPEN,kernel)
    # cv2.imshow('container',container)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

    grayc = cv2.cvtColor(container,cv2.COLOR_BGR2GRAY)
    blurredc = cv2.GaussianBlur(grayc, (5,5), 0)
    threshc = cv2.threshold(blurredc, 3, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    cntsc = cv2.findContours(threshc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsc = cntsc[0] if imutils.is_cv2() else cntsc[1]
    shaper=ShapeDetector()

    board_objects = []		# List to store output of board -- DO NOT CHANGE VARIABLE NAME
    board1_objects = []
    container_objects=[]
    output_list = []

    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX=int(M["m10"] / M["m00"])
        cY=int(M["m01"] / M["m00"])
        #print(cX,cY)
        z=dig(cX,cY)
        colour=colorfun(cX,cY,board)
        shape=shaper.detect(c)
        perimeter = cv2.arcLength(c,True)

        temp=(z,colour,shape)
        board_objects.append(temp)
        temp1=(z,colour,shape,perimeter)
        board1_objects.append(temp1)

    for c in cntsc:
        M = cv2.moments(c)
        cX=int(M["m10"] / M["m00"])
        cY=int(M["m01"] / M["m00"])
        #print(cX,cY)
        z=digc(cX,cY)
        #print z
        colour=colorfun(cX,cY,container)
        shape=shaper.detect(c)
        perimeter = cv2.arcLength(c,True)

        temp=(z,colour,shape,perimeter)
        container_objects.append(temp)

    #print board_objects
    #print container_objects

    board_objects=sorted(board_objects)
    container_objects=sorted(container_objects)
    board1_objects=sorted(board1_objects)
    #print board_objects
    #print container_objects

    #ans=[]
    for i in range (0,9):
        temp=(board1_objects[i][0],0)
        for j in range (0,len(container_objects)):
            if board1_objects[i][1]==container_objects[j][1] and board1_objects[i][2]==container_objects[j][2] and abs(board1_objects[i][3]-container_objects[j][3])<14:
                temp = (board1_objects[i][0],container_objects[j][0])
                break
        output_list.append(temp)
    print(output_list)

    return board_objects, output_list
'''
Below part of program will run when ever this file (task1_main.py) is run directly from terminal/Idle prompt.

'''
if __name__ == '__main__':


    board_filepath = "test_images/board_4.jpg"    			# change filename of board provided to you
    container_filepath = "test_images/container_5.jpg"		# change filename of container as required for testing

    main(board_filepath,container_filepath)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
