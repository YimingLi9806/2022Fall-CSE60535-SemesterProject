import cv2
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)

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

def findPicDescriptor(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    #FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  #2
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matchList = []
    finalVal = -1


    try:        
        for des in desList:
            matches = flann.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
     #print(matchList)
    if len(matchList)!=0:
        if max(matchList)>20:
            finalVal = matchList.index(max(matchList))
    return finalVal


desList = findPicDescriptor(images)
#print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2,desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    else:
        cv2.putText(imgOriginal, "Not one of the five buildings", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)


