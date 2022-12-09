import cv2
import numpy as np
import os
import math

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=3, nOctaveLayers=3)

path = 'ImagesQuery'
images = []
classNames = []
myList = os.listdir(path)


print('total detected', len(myList))
for c1 in myList:
    imgCur = cv2.imread(f'{path}/{c1}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(c1)[0])
print(classNames)

def findDes(images):
    desList = []
    for img in images:
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp,des = surf.detectAndCompute(grayImg, None)
        desList.append(des)
    return desList

def findID(img, desList):
    kp2, des2 = surf.detectAndCompute(img, None)
    bf = cv2.BFMatcher(crossCheck=True)
    matchList = []
    finalVal = -1
    
    try:        
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
     #print(matchList)
    if len(matchList)!=0:
        if max(matchList)>15:
            finalVal = matchList.index(max(matchList))
    return finalVal


desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2,desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)


