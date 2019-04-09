# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:03:41 2018

@author: amounesi
"""


# Breaking down the images for the training purposes



import cv2 

i=0
cap = cv2.VideoCapture('images/Trailing3.MKV') # images/go_r_1.MP4 ,,, images/Trailing3.MKV

while(cap.isOpened()):
    
    i=i+1
    ret, frame = cap.read()

    if ret == True:
        if i > 1700: 

            
# how to zoom                zoomed700X700 = frame [1220:1920, 190:890 , :]

                
                
                cv2.imshow('image', frame )
                cv2.imwrite('images/picout/output'+str(i)+'.jpg', frame)

                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
    else:
        break


cap.release()
cv2.destroyAllWindows()
