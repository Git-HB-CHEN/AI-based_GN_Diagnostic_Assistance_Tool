import cv2
import numpy as np

def FindContours(Image1C):
    _, binary = cv2.threshold(Image1C.astype(np.uint8),127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return contours