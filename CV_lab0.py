import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

while(cap.isOpened()):
    flag, cadre = cap.read()
    out.write(cadre)
    cv2.imshow('cadre',cadre)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('cadre')
cap = cv2.VideoCapture('output.avi')

while(cap.isOpened()):
       
    flag, cadre = cap.read()
    if flag:
        gray = cv2.cvtColor(cadre, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(gray,(200,100),(400,200),(0,255,0),3)
        cv2.line(gray,(0,0),(640,300),(100,10,200),5)

        cv2.imshow('gray',gray)
        cv2.waitKey(44)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()