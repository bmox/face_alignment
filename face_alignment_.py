import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_face_detection= mp.solutions.face_detection
cap=cv2.VideoCapture(0)
face_detection=mp_face_detection.FaceDetection(min_detection_confidence=0.5)

while True:
    ret,img= cap.read()
    img=cv2.flip(img,1)
    h,w,_ = img.shape
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=face_detection.process(img_rgb)
    if results.detections:
        for detection in results.detections:
            #e1
            x1=int(detection.location_data.relative_keypoints[0].x*w)
            y1=int(detection.location_data.relative_keypoints[0].y*h)
            #e2
            x2=int(detection.location_data.relative_keypoints[1].x*w)
            y2=int(detection.location_data.relative_keypoints[1].y*h)

            p1=np.array([x1,y1])
            p2=np.array([x2,y2])
            p3=np.array([x2,y1])
            
            #find distance
            d_eyes = np.linalg.norm(p1-p2)
            l1=np.linalg.norm(p1-p3)
            #find angle
            angle =degrees(acos(l1/d_eyes))

            #find posative or negative
            if y1<y2:
                angle=-angle
            #rotate image
            M= cv2.getRotationMatrix2D((w//2,h//2), -angle, 1)
            aligned_image=cv2.warpAffine(img, M, (w,h))
            cv2.imshow("Aligned Image",aligned_image)
            #display
            cv2.putText(img, "p1", (x1-60,y1), 1, 1.5,(0,255,0),2)
        
            cv2.putText(img, "p2", (x2+30,y2), 1, 1.5,(0,128,255),2)
            cv2.putText(img, "p3", (x2+60,y1), 1, 1.5,(255,0,255),2)
            cv2.putText(img, str(int(angle)), (x1-35,y1+15), 1, 1,(0,255,0),2)

            #line
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)      
            cv2.line(img,(x1,y1),(x2,y1),(211,0,128),2)    
            cv2.line(img,(x2,y2),(x2,y1),(0,128,255),2)  

            #circle

            cv2.circle(img,(x1,y1),5,(0,255,0),-1)
            cv2.circle(img,(x2,y2),5,(0,128,255),-1)
            cv2.circle(img,(x2,y1),5,(211,0,148),-1)

            #face_detection again
            aligne_img_rgb=cv2.cvtColor(aligned_image,cv2.COLOR_BGR2RGB)
            results2=face_detection.process(aligne_img_rgb)
            if results2.detections:
                for detection in results2.detections:
                    xmin=int(detection.location_data.relative_bounding_box.xmin*w)
                    ymin=int(detection.location_data.relative_bounding_box.ymin*h)
                    width=int(detection.location_data.relative_bounding_box.width*w)
                    height=int(detection.location_data.relative_bounding_box.height*h)

                    if xmin <0 or ymin < 0:
                        continue
                    aligned_face=aligned_image[ymin:ymin+height,xmin:xmin+width]
                    cv2.imshow("Aligned face",aligned_face)

    cv2.imshow("Display",img)
    cv2.waitKey(1)
