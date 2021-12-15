import cv2 as cv
import numpy as np

#Load Image
vid = cv.VideoCapture('video.mp4')
floor=0
hold=0
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        print("Can not receive frame. Exiting . . . ")
        break
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret, thres = cv.threshold (gray, 100, 255,cv.THRESH_BINARY)

    #Kernel Gabor
    def get_gabor_kernel(size,sigma,lamda,theta,gamma):
        center = (int)(size/2)
        gabor_kernel = np.zeros((size,size))
        for x in range(-center, center+1, 1):
            for y in range(-center, center+1, 1):
                x_aks = x* np.cos(theta) + y* np.sin(theta)
                y_aks = -x* np.sin(theta) + y* np.cos(theta)
                gabor_kernel[x+center,y+center] = np.exp(-(x_aks ** 2 + gamma ** 2 * y_aks ** 2)/(2 * sigma ** 2)) * np.cos(2 * np.pi * x_aks / lamda)
        return gabor_kernel
    kernel_gab = get_gabor_kernel(11,1,5,135,0.25)
    gabor_filter = cv.filter2D(src=thres, ddepth=-1, kernel=kernel_gab)

    #Ambil ROI
    crop = gabor_filter[350:450, 0:540]
    ret, crop = cv.threshold (crop, 100, 255,cv.THRESH_BINARY_INV)
    mask = np.zeros_like(gabor_filter)
    mask[350:450,0:540]=crop[:,:]

    #Hitung Blob
    output=cv.connectedComponentsWithStats(mask,8,cv.CV_32S)
    (jml_label, label, stats, centroid)=output
    jml_blob=0
    for i in range(0,jml_label) :
        if stats[i,cv.CC_STAT_AREA]>600 :
            jml_blob = jml_blob+1

    if hold==0 and jml_blob>=2 :
        floor = floor+1
        hold = 1
    elif hold==1 and jml_blob<=1 :
        hold = 0

    #Hasil
    cv.imshow('Original', frame)
    #cv.imshow('Threshold', thres)
    #cv.imshow('Gabor', gabor_filter)
    cv.imshow('Region of Interest', mask)
    print("Jumlah Blob = ",jml_blob)
    print("Jumlah garis lantai terlewati ",floor)

    keyboard = cv.waitKey(38)
    if keyboard == 'q' or keyboard == 27:
        break
cv.destroyAllWindows()
