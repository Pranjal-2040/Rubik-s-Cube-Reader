import numpy as np
import cv2
import os
n=len(os.listdir("input"))
#print(n)
# Load image and keep a copy
input_address="input\Image{}.PNG"
for I in range(1,n+1):
 image = cv2.imread(input_address.format(I))
 orig_image = image.copy()
 # cv2.imshow('Original Image', orig_image)
 # cv2.waitKey(0) 

 # Grayscale and binarize
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
 gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
 gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

 gray = cv2.adaptiveThreshold(gray,20,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,0)

 #  Find contours 
 contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
  # for c in contours:
 #     x,y,w,h = cv2.boundingRect(c)
 #     cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)    
 #     cv2.imshow('Bounding Rectangle', orig_image)

 # cv2.waitKey(0) 
 # Iterate through each contour and compute the bounding rectangle
 i = 0
 contour_id = 0
    #print(len(contours))
 number = 0
 blob_colors = []
 for contour in contours:
        A1 = cv2.contourArea(contour)
        # print(A1)
        contour_id = contour_id + 1

        if A1 < 7500 and A1 >4500:
            
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(contour)
            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 2000:
                #if cv2.ma
                number = number + 1
                x, y, w, h = cv2.boundingRect(contour)
 
             
                val = (50*y) + (10*x)
                blob_color = np.array(cv2.mean(orig_image[y:y+h,x:x+w])).astype(int)
                cv2.drawContours(orig_image,[contour],0,(226, 43, 138),2)
                cv2.drawContours(orig_image, [approx], 0, (226, 43, 138), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)
                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)
                blob_colors.append(blob_color)
 if len(blob_colors) > 0:
        blob_colors = np.asarray(blob_colors)
        blob_colors = blob_colors[blob_colors[:, 4].argsort()]
 # print(number)

 face_colour = np.array([0,0,0,0,0,0,0,0,0])
 if len(blob_colors) == 9:
        #print(blob_colors)
    for i in range(9):
            #print(blob_colors[i])
            if blob_colors[i][0] > 120 and blob_colors[i][1] > 120 and blob_colors[i][2] > 100:
                blob_colors[i][3] = 1  #White
                face_colour[i] = 1
            elif blob_colors[i][0] < 100 and blob_colors[i][1] > 120 and blob_colors[i][2] > 120 and np.abs(blob_colors[i][1]-blob_colors[i][2])<30:
                blob_colors[i][3] = 2  #Yellow
                face_colour[i] = 2
            elif blob_colors[i][0] > blob_colors[i][1] and blob_colors[i][1] > blob_colors[i][2]:
                blob_colors[i][3] = 3 #Blue
                face_colour[i] = 3
            elif blob_colors[i][1] > blob_colors[i][0] and blob_colors[i][1] > blob_colors[i][2] and np.abs(blob_colors[i][0] - blob_colors[i][2]) < 30:
                blob_colors[i][3] = 4   #Green
                face_colour[i] = 4
            elif blob_colors[i][2] > blob_colors[i][0] and blob_colors[i][2] > blob_colors[i][1] and np.abs(blob_colors[i][0] - blob_colors[i][1]) < 30 and blob_colors[i][0] < 80:
                blob_colors[i][3] = 5 #Red
                face_colour[i] = 5
            elif blob_colors[i][1] < blob_colors[i][2] and blob_colors[i][0] < blob_colors[i][1] and blob_colors[i][2] > 120:
                blob_colors[i][3] = 6  #Orange
                face_colour[i] = 6
 #print(face_colour)
 output_address="Output\Output_Image{}.txt"
 f= open(output_address.format(I),"w")

 colour1=str(face_colour[:3])

 colour2=str(face_colour[3:6])
 colour3=str(face_colour[6:9])
 f.write(colour1)
 f.write('\n')
 f.write(colour2)
 f.write('\n')
 f.write(colour3)
 f.close()

# # cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 255, 255), 2)   
# cv2.imshow('gray',orig_image)
# cv2.waitKey(0)
# Iterate through each contour and compute the approx contour
