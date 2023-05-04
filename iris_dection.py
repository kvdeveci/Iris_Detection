import cv2 as cv
import numpy as np 
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh

left_eyes =[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
# right eyes indices
right_eyes =[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
  
left_iris = [474,475,476,477]
right_iris = [469,470,471,472]

cap = cv.VideoCapture(0)#it is camerea
with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    while True:
            ret,frame = cap.read()
            if not ret:
                    break
            
            frame = cv.flip(frame,1)
            rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            img_h,img_W = frame.shape[:2]
            resolt = face_mesh.process(rgb_frame)
            if resolt.multi_face_landmarks:
                   mesh_point =np.array([np.multiply([p.x,p.y],[img_W,img_h]).astype(int)for p in resolt.multi_face_landmarks[0].landmark])  
                   (l_cx, l_cy),l_radious =  cv.minEnclosingCircle(mesh_point[left_iris])
                   (r_cx, r_cy),r_radious =  cv.minEnclosingCircle(mesh_point[right_iris])
                   centre_left = np.array([l_cx,l_cy],dtype=np.int32)
                   centre_rigth = np.array([r_cx,r_cy],dtype=np.int32)
                   cv.circle(frame,centre_left,int(l_radious),(0,0,255),1,cv.LINE_AA)
                   cv.circle(frame,centre_rigth,int(r_radious),(0,0,255),1,cv.LINE_AA)

                
            cv.imshow('img',frame)            
            key = cv.waitKey(1)
            if key == ord('q'):
                    break
cap.release()
cv.destroyAllWindows()       