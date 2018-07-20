from collections import deque

import numpy as np
import cv2

from constants import * 
from misc      import *
from draw      import *

#-------------------------------------------------------------------

def init_search(video, fname):

#-------------------------------------------------------------------
    cap = cv2.VideoCapture(video)
    
    arrTLight = deque(maxlen=FRAME_LENGTH)
    approvedTLight = []

    FrameCount = 0

    arrFrameRed   = deque(maxlen=2*FRAME_LENGTH)
    arrFrameGreen = deque(maxlen=2*FRAME_LENGTH)
    arrFramePrev  = deque(maxlen=2*FRAME_LENGTH)

#-------------------------------------------------------------------

    while(cap.isOpened()):

#--------------------------------------------------------------------
        ret, Frame = cap.read()
        if Frame is None:
            if len(approvedTLight) == 0:
                print(fname, -1)
            break
        ret, Frame = cap.read()
        if Frame is None:
            if len(approvedTLight) == 0:
                print(fname, -1)
            break

#-------------------------------------------------------------------

        lastFrame  = FrameCount
        FrameCount = FrameCount + 2
        FrameHeigth, FrameWidth, _ = Frame.shape

#-------------------------------------------------------------------

        BoxHeigth, BoxWidth = int(0.7 * FrameHeigth), int(0.8 * FrameWidth)
        FrameBox = Frame[0 : BoxHeigth, int((FrameWidth - BoxWidth) / 2) : int((FrameWidth + BoxWidth) / 2)]

        FrameBlur  = cv2.GaussianBlur(FrameBox, (7, 7), 0.5)
        FrameHSV   = cv2.cvtColor(FrameBlur, cv2.COLOR_BGR2HSV)

        maskRed   = cv2.inRange(FrameHSV, RED_MR, RED_R) + cv2.inRange(FrameHSV, RED_L, RED_ML)
        maskGreen = cv2.inRange(FrameHSV, GREEN_L, GREEN_R)

#-------------------------------------------------------------------

        if len(arrFramePrev) < 1:
            arrFramePrev.appendleft(FrameBlur)

        FrameDelta = cv2.absdiff(arrFramePrev[len(arrFramePrev) - 1], FrameBlur)
        arrFramePrev.appendleft(FrameBlur)

#-------------------------------------------------------------------        

        maskRedDelta  = np.zeros(BoxHeigth*BoxWidth, dtype = "uint8").reshape(BoxHeigth,BoxWidth)
        
        if len(arrFrameRed) > 1 :
            maskRedDelta = arrFrameRed[len(arrFrameRed) - 1] - maskRed
        arrFrameRed.appendleft(maskRed)

        maskGreenDelta = np.zeros(BoxHeigth*BoxWidth, dtype = "uint8").reshape(BoxHeigth,BoxWidth)

        if len(arrFrameGreen) > 1 :
            maskGreenDelta = maskGreen - arrFrameGreen[len(arrFrameGreen) - 1]
        arrFrameGreen.appendleft(maskGreen)

#-------------------------------------------------------------------

        maskRedDelta        = cv2.erode(maskRedDelta, None, iterations=1)
        maskRedDelta        = cv2.dilate(maskRedDelta, None, iterations=3)
        ret, maskRedDelta   = cv2.threshold(maskRedDelta, 120, 250, cv2.THRESH_BINARY)

        maskGreenDelta      = cv2.erode(maskGreenDelta, None, iterations=1)
        maskGreenDelta      = cv2.dilate(maskGreenDelta, None, iterations=3)
        ret, maskGreenDelta = cv2.threshold(maskGreenDelta, 120, 250, cv2.THRESH_BINARY)

#-------------------------------------------------------------------

        RedContours   = cv2.findContours(maskRedDelta  , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        GreenContours = cv2.findContours(maskGreenDelta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

#-------------------------------------------------------------------

        arrRedCircle   = getcircles(RedContours, BoxHeigth)
        arrGreenCircle = getcircles(GreenContours, BoxHeigth)

#-------------------------------------------------------------------

        TLights = []
        for RedCircle in arrRedCircle:
            ((RedCircleCnt_X, RedCircleCnt_Y), RedContourRad) = RedCircle[0]

            RedCircleCnt_X = int(RedCircleCnt_X)
            RedCircleCnt_Y = int(RedCircleCnt_Y)
            
            Top = 0
            Left = 0
            Right = FrameDelta.shape[1]
            Bottom = FrameDelta.shape[0]
            
            CircleOffset = OFFSET + RedContourRad
            
            if Top < int(RedCircleCnt_Y - CircleOffset):
                Top = int(RedCircleCnt_Y - CircleOffset)
            
            if Left < int(RedCircleCnt_X - (X_MULT_OFFSET*CircleOffset)):
                Left = int(RedCircleCnt_X - (X_MULT_OFFSET*CircleOffset))
            
            if Right > int(RedCircleCnt_X + (X_MULT_OFFSET*CircleOffset)):
                Right = int(RedCircleCnt_X + (X_MULT_OFFSET*CircleOffset))
            
            if Bottom > int(RedCircleCnt_Y + CircleOffset):
                Bottom = int(RedCircleCnt_Y + CircleOffset)
            
            RedBox = FrameDelta[Top:Bottom,Left:Right,:]

            RedGreenDelta = np.sum(RedBox) - (360*RedContourRad*RedContourRad)
            TotalDelta = RedGreenDelta / (RedBox.shape[0]*RedBox.shape[1])
            
            for GreenCircle in arrGreenCircle:

                DeltaX = RedCircle[0][0][0] - GreenCircle[0][0][0]
                DeltaY = RedCircle[0][0][1] - GreenCircle[0][0][1]
                DeltaR = RedCircle[0][1]/GreenCircle[0][1]
                RedRad = RedCircle[0][1]
                MaxDeltaY = -5*(RedRad+GreenCircle[0][1])/2
                if MaxDeltaY < -170:
                    MaxDeltaY = -170
                if DeltaR > 0.4 and DeltaR < 2.5:
                    if DeltaX < (RedRad/2) and DeltaX > (-RedRad/2):
                        if DeltaY > MaxDeltaY and DeltaY < (-(RedRad+GreenCircle[0][1])/4) and DeltaY < -7:
                            if TotalDelta < 33:
                                new_TLight = (RedCircle[0], GreenCircle[0],TotalDelta)
                                TLights.append(new_TLight)

#-------------------------------------------------------------------

        approve_tlight(TLights, arrTLight, approvedTLight, FrameCount, lastFrame)

        arrTLight.appendleft(TLights)

#------------------------------------------------------------------
        
        draw_all(TLights, arrRedCircle, arrGreenCircle, approvedTLight, FrameBox)

#-------------------------------------------------------------------

        cv2.imshow("images", FrameBox)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(approvedTLight) > 0:
            print(fname, lastFrame)
            break

#-------------------------------------------------------------------

    cap.release()