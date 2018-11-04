import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import time
import math

dir = os.listdir('dataset')
im1 = cv2.imread('first.JPG')

detector = cv2.BRISK_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING,True)
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
(kps1, descs1) = detector.detectAndCompute(gray1, None)
fil = open("result_BRISK", "w")
t = 0
size = 0
counter = 0
acc = 0

for file in dir:
    img = cv2.imread('dataset/' + file)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)
    start = time.clock() 
    matches = bf.match(descs1, descs2) 
    end = time.clock()
    f = end-start

    good = sorted(matches, key=lambda x: x.distance)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_low = (0, 0, 23)
    color_high = (87, 28, 255)

    low = np.array(color_low, np.uint8)
    high = np.array(color_high, np.uint8)

    thresh = cv2.inRange(hsv, low, high)
    moments = cv2.moments(thresh, 1)
    x_moment = moments['m10']
    y_moment = moments['m01']
    area = moments['m00']

    x = int(x_moment / area)
    y = int(y_moment / area)
   
    points = [0, 0]
    for m in good:
        points[0] += kps2[m.trainIdx].pt[0]
        points[1] += kps2[m.trainIdx].pt[1]
    x1 = int(points[0]/len(good))
    y1 = int(points[1]/len(good))

    dist = math.sqrt((x-x1)**2+(y-y1)**2)
    c = len(good)/len(kps1)
    t = t + f
    counter += 1
    acc = acc + c
    size = size + os.path.getsize('dataset/' + file)

    fil.write('\nname - '+ file + '\ntime = ' + str(f*1000000/os.path.getsize('dataset/' + file)) + '\nkeyp 1 = ' + str(len(kps1)) + '\nkeyp 2 = ' + str(len(kps2))
    + '\nmatches(soterd) = ' + str(len(good)) + '\naccuracy = ' + str(c) + '\nAVG_time = ' + str(t*1000000/size) + '\nAVG_accuracy = ' + str(acc/counter) + '\ndistance = ' + str(dist))

    print('name- ', file)
    print('time = ' + str(f*1000000/os.path.getsize('dataset/' + file)))
    print('keyp 1 = ' + str(len(kps1)))
    print('keyp 2 = ' + str(len(kps2)))
    print('matches(soterd) = ' + str(len(good)))
    print('accuracy = ' + str(c))
    print('AVG_time = ' + str(t*1000000/size))
    print('AVG_accuracy = ' + str(acc/counter))
    print('distance = ' + str(dist))

fil.close()



'''
img = cv2.imread('dataset/DSC_0130.JPG')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
color_low = (0, 0, 23)
color_high = (87, 28, 255)

hsv_min = np.array(color_low, np.uint8)
hsv_max = np.array(color_high, np.uint8)

thresh = cv2.inRange(hsv, hsv_min, hsv_max)
moments = cv2.moments(thresh, 1)
x_moment = moments['m10']
y_moment = moments['m01']
area = moments['m00']

x = int(x_moment / area)
y = int(y_moment / area)
cv2.circle(img, (x,y), 40, (255, 255, 255), -1)

plt.imshow(img), plt.show() '''

'''
M = cv2.moments(thresh)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv2.circle(img, (cX, cY), 50, (255, 255, 255), -1)
plt.imshow(img), plt.show()
'''
'''
img = cv2.imread('dataset/DSC_0006.JPG')
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(kps2, descs2) = detector.detectAndCompute(gray2, None)
matches = bf.match(descs1,descs2) 
good = sorted(matches,key = lambda x : x.distance)

points = [0, 0]
for m in good:
    points[0] += kps2[m.trainIdx].pt[0]
    points[1] += kps2[m.trainIdx].pt[1]
x1= int(points[0]/len(good))
y1= int(points[1]/len(good))

cv2.circle(img, (x1,y1), 40, (255, 255, 255), -1)

plt.imshow(img), plt.show()
#im3 = cv2.drawMatches(gray1, kps1, gray2, kps2,good[:10], None, flags=2)
#fil.close()
#plt.imshow(im3), plt.show() 
#cv2.waitKey(0) '''