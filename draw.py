import cv2

def draw_all(TLights, arrRedCircle, arrGreenCircle, approvedTLight, FrameBox):
    for TLight in TLights:
        cv2.circle(FrameBox, (int(TLight[0][0][0]), int(TLight[0][0][1])), int(20), (255, 0, 255), 2)

    for RedCircle in arrRedCircle:
        cv2.circle(FrameBox, (int(RedCircle[0][0][0]), int(RedCircle[0][0][1])), int(RedCircle[0][1]), (0, 0, 255), 2)

    for GreenCircle in arrGreenCircle:
        cv2.circle(FrameBox, (int(GreenCircle[0][0][0]), int(GreenCircle[0][0][1])), int(GreenCircle[0][1]), (0, 255, 0), 2)

    for TLight in approvedTLight:
        cv2.circle(FrameBox, (int(TLight[0][0][0]), int(TLight[0][0][1])), int(TLight[0][1]), (255, 0, 0), 2)
        cv2.circle(FrameBox, (int(TLight[1][0][0]), int(TLight[1][0][1])), int(TLight[1][1]), (255, 255, 0), 2)