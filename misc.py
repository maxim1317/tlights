import math
import cv2

def dist(p1,p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 )

def approve_tlight(TLights, arrTLight, approvedTLight, FrameCount, lastFrame):
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

def getcircles(Contours, BoxHeigth):
    area = 0
    arrCircle = []

    for Contour in Contours:
        Circle = cv2.minEnclosingCircle(Contour)

        ((Contour_x, Contour_y), ContourRad) = Circle

        MinRad = 2 + (Contour_x/BoxHeigth) * 3
        MaxRad = 35 + (Contour_x/BoxHeigth) * 30

        if ContourRad > MinRad and ContourRad < MaxRad:
            new_circle = (Circle, area)
            arrCircle.append(new_circle)
    return arrCircle