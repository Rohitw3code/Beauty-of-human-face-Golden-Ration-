import cv2
import mediapipe as mp
import math
import random as rd
import numpy as np
from cvzone.HandTrackingModule import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

capture = cv2.VideoCapture(0)
frameWidth = 720
frameHeight = 720
capture.set(3, frameWidth)
capture.set(4, frameHeight)


class GoldenRatio():
    def __init__(self):
        self.phi = 1.618
        self.minLips = 100
        self.maxLips = 0
        self.minEyeBrow = 100
        self.maxEyeBrow = 0
        self.minNose = 100
        self.maxNose = 0
        self.maxFace = 0
        self.minFace = 100

    def distance(self,p1,p2):
        return math.hypot(p1[0]-p2[0],p1[1]-p2[1])

    def percent(self,gr):
        return 100 - abs((abs(self.phi - gr)/self.phi)*100)

    def nose(self,landmark):
        coodinates = np.array(landmark).reshape((-1, 1, 2))
        upper = [64,294]
        lower = [97,326]
        a = self.distance(coodinates[upper[0]][0],coodinates[upper[1]][0])
        b = self.distance(coodinates[lower[0]][0],coodinates[lower[1]][0])
        gr = a/b
        perc = self.percent(gr)
        if self.minNose > perc:
            self.minNose = perc
        if self.maxNose < perc:
            self.maxNose = perc
        return int((self.minNose + self.maxNose) / 2)


    def eyebrow(self,landmark):
        coodinates = np.array(landmark).reshape((-1, 1, 2))
        right = [55,52,43]
        left = [285,282,276]
        ra = self.distance(coodinates[right[0]][0],coodinates[right[1]][0])
        rb = self.distance(coodinates[right[1]][0],coodinates[right[2]][0])

        la = self.distance(coodinates[left[0]][0],coodinates[left[1]][0])
        lb = self.distance(coodinates[left[1]][0],coodinates[left[2]][0])

        rgr = ra/rb
        lgr = la/lb
        gr = (rgr+lgr)/2
        perc = self.percent(gr)
        if self.minEyeBrow > perc:
            self.minEyeBrow = perc
        if self.maxEyeBrow < perc:
            self.maxEyeBrow = perc
        return int((self.minEyeBrow + self.maxEyeBrow) / 2)

    def face(self,landmark):
        v = [10,152]
        h = [234,454]
        coodinates = np.array(landmark).reshape((-1, 1, 2))
        vdist = self.distance(coodinates[v[0]][0],coodinates[v[1]][0])
        hdist = self.distance(coodinates[h[0]][0],coodinates[h[1]][0])
        gr = vdist/hdist
        perc = self.percent(gr)
        if self.minFace > perc:
            self.minFace = perc
        if self.maxFace < perc:
            self.maxFace = perc
        return int((self.minFace + self.maxFace) / 2)



    def lips(self,landmark):
        v = [0,13]
        h = [14,17]
        coodinates = np.array(landmark).reshape((-1, 1, 2))
        vdist = self.distance(coodinates[v[0]][0],coodinates[v[1]][0])
        hdist = self.distance(coodinates[h[0]][0],coodinates[h[1]][0])
        gr = hdist/vdist
        perc = self.percent(gr)
        if self.minLips > perc:
            self.minLips = perc
        if self.maxLips < perc:
            self.maxLips = perc
        return int((self.minLips + self.maxLips) / 2)


goldenRatio = GoldenRatio()

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())

        landmarks = []
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y

                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    landmarks.append([relative_x, relative_y])

        a = goldenRatio.lips(landmarks)
        b = goldenRatio.eyebrow(landmarks)
        c = goldenRatio.nose(landmarks)
        d = goldenRatio.face(landmarks)
        goldenvalue = (a+b+c+d)/4

        cv2.rectangle(image, (30,10), (400,80), (255,255,255), -1)
        cv2.putText(image,"Beautiful  : "+str(round(goldenvalue,2))+" %",(50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2, cv2.LINE_AA)


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()