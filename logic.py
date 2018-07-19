from collections import deque

import numpy as np
import cv2

from constants import * 
from misc      import *

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
        ret, maskRedDelta   = cv2.threshold(maskRedDelta, 127, 250, cv2.THRESH_BINARY)

        maskGreenDelta      = cv2.erode(maskGreenDelta, None, iterations=1)
        maskGreenDelta      = cv2.dilate(maskGreenDelta, None, iterations=3)
        ret, maskGreenDelta = cv2.threshold(maskGreenDelta, 127, 250, cv2.THRESH_BINARY)

#-------------------------------------------------------------------

        RedContours   = cv2.findContours(maskRedDelta  , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        GreenContours = cv2.findContours(maskGreenDelta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

#-------------------------------------------------------------------

        area = 0
        arrRedCircle = []

        for RedContour in RedContours:
            RedCircle = cv2.minEnclosingCircle(RedContour)

            ((RedContour_x, RedContour_y), RedContourRad) = RedCircle

            MinRad = 2 + (RedContour_x/BoxHeigth) * 3
            MaxRad = 35 + (RedContour_x/BoxHeigth) * 30

            if RedContourRad > MinRad and RedContourRad < MaxRad:
                new_circle = (RedCircle, area)
                arrRedCircle.append(new_circle)

        arrGreenCircle = []

        for GreenContour in GreenContours:
            GreenCircle = cv2.minEnclosingCircle(GreenContour)

            ((GreenContour_x, GreenContour_y), GreenContourRad) = GreenCircle

            MinRad = 2 + (GreenContour_x/BoxHeigth) * 3
            MaxRad = 35 + (GreenContour_y/BoxHeigth) * 30

            if GreenContourRad > MinRad and GreenContourRad < MaxRad:
                new_circle = (GreenCircle, area)
                arrGreenCircle.append(new_circle)

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
            
            CircleOFFSET = OFFSET + RedContourRad
            
            if Top < int(RedCircleCnt_Y - CircleOFFSET):
                Top = int(RedCircleCnt_Y - CircleOFFSET)
            
            if Left < int(RedCircleCnt_X - (X_MULT_OFFSET*CircleOFFSET)):
                Left = int(RedCircleCnt_X - (X_MULT_OFFSET*CircleOFFSET))
            
            if Right > int(RedCircleCnt_X + (X_MULT_OFFSET*CircleOFFSET)):
                Right = int(RedCircleCnt_X + (X_MULT_OFFSET*CircleOFFSET))
            
            if Bottom > int(RedCircleCnt_Y + CircleOFFSET):
                Bottom = int(RedCircleCnt_Y + CircleOFFSET)
            
            red_crop = FrameDelta[Top:Bottom,Left:Right,:]

            total_move_GREEN_Red = np.sum(red_crop) - (380*RedContourRad*RedContourRad)
            total_move = total_move_GREEN_Red / (red_crop.shape[0]*red_crop.shape[1])
            
            for GreenCircle in arrGreenCircle:

                dif_x = RedCircle[0][0][0] - GreenCircle[0][0][0]
                dif_y = RedCircle[0][0][1] - GreenCircle[0][0][1]
                dif_r = RedCircle[0][1]/GreenCircle[0][1]
                radius_red = RedCircle[0][1]
                max_dist_y = -5*(radius_red+GreenCircle[0][1])/2
                if max_dist_y < -170:
                    max_dist_y = -170
                if dif_r > 0.4 and dif_r < 2.5:
                    if dif_x < (radius_red/2) and dif_x > (-radius_red/2):
                        if dif_y > max_dist_y and dif_y < (-(radius_red+GreenCircle[0][1])/4) and dif_y < -7:
                            if total_move < 33:
                                new_TLight = (RedCircle[0], GreenCircle[0],total_move)
                                TLights.append(new_TLight)

#-------------------------------------------------------------------

        for TLight in TLights:
            true_TLight = 0
            Frame_delta = 1
            for last_TLights in arrTLight:
                Frame_delta = Frame_delta + 1
                for last_TLight in last_TLights:
                    distance_red = dist(TLight[0][0],last_TLight[0][0])
                    distance_green = dist(TLight[1][0],last_TLight[1][0])
                    if distance_red < (TLight[0][1]*0.5) and distance_green < (TLight[1][1]*0.5):
                        true_TLight = true_TLight + 1
                        if (FrameCount - Frame_delta) < lastFrame:
                            lastFrame = FrameCount - Frame_delta
                        break
            if true_TLight > 1:
                approvedTLight.append(TLight)
                break

        arrTLight.appendleft(TLights)

#-------------------------------------------------------------------

        for TLight in TLights:
            cv2.circle(FrameBox, (int(TLight[0][0][0]), int(TLight[0][0][1])), int(20),
                                (255, 0, 255), 2)

        for RedCircle in arrRedCircle:
            cv2.circle(FrameBox, (int(RedCircle[0][0][0]), int(RedCircle[0][0][1])), int(RedCircle[0][1]),
                                (0, 0, 255), 2)

        for GreenCircle in arrGreenCircle:
            cv2.circle(FrameBox, (int(GreenCircle[0][0][0]), int(GreenCircle[0][0][1])), int(GreenCircle[0][1]),
                                (0, 255, 0), 2)

        for TLight in approvedTLight:
            cv2.circle(FrameBox, (int(TLight[0][0][0]), int(TLight[0][0][1])), int(TLight[0][1]),
                                (255, 0, 0), 2)
            cv2.circle(FrameBox, (int(TLight[1][0][0]), int(TLight[1][0][1])), int(TLight[1][1]),
                                (255, 255, 0), 2)

#-------------------------------------------------------------------

        cv2.imshow("images", FrameBox)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(approvedTLight) > 0:
            print(fname, lastFrame)
            break

#-------------------------------------------------------------------

    cap.release()