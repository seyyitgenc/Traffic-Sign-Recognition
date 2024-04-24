import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as rt
import time

modelPath = 'C:\projects_ai\Traffic-Sign-Recognition\sign.onnx'
# model = keras.models.load_model("Detection\model.h5")

# Load the ONNX model
sess = rt.InferenceSession(modelPath)

def returnRedness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v

def returnBlueness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return u

def threshold(img,T=150):
	_, img = cv2.threshold(img,T,255,cv2.THRESH_BINARY)
	return img 

def findContour(img):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def findBiggestContour(contours):
	m = 0
	c = [cv2.contourArea(i) for i in contours]
	return contours[c.index(max(c))]

def boundaryBox(img,contours):
	x, y, w, h = cv2.boundingRect(contours)
	img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	sign = img[y:(y+h) , x:(x+w)]
	return img, sign

def preprocessingImageToClassifier(image=None,imageSize=32,mu=89.77428691773054,std=70.85156431910688):
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image,(imageSize,imageSize))
	image = cv2.equalizeHist(image)
	image = image.astype(np.float32)  # Convert to float32
	image = image/255
	image = image.reshape(1,imageSize,imageSize,1)
	return image

def predict(sign):
	img = preprocessingImageToClassifier(sign, imageSize=32)
	input_name = sess.get_inputs()[0].name
	probabilities = sess.run(None, {input_name: img})
	predicted_class = np.argmax(probabilities)
	return predicted_class, np.max(probabilities)

#--------------------------------------------------------------------------
labelToText = { 0:"Stop",
    			1:"Turn Left",
    			2:"Turn Right",
    			3:"Forward"}
# cap=cv2.VideoCapture('C:/projects_ai/Traffic-Sign-Recognition/Test Video/test2.mp4')
cap=cv2.VideoCapture(0)

swap : bool = True

while(True):
	_, frame = cap.read()
	redness = returnRedness(frame)
	blueness = returnBlueness(frame) 

	blue_region = threshold(blueness)
	red_region = threshold(redness)
	try:
		contours = 0
		# if swap:
		# 	contours = findContour(blue_region)
		# 	swap = False
		# else:
		contours = findContour(red_region)
			# swap = True
		big = findBiggestContour(contours)
		if cv2.contourArea(big) > 3000:
			img,sign = boundaryBox(frame,big)
			predicted_class, probability = predict(sign)
			print(predicted_class)
			if(probability * 100 > 20):
				print(f"Now, I see: {labelToText[predicted_class]} with probability: {probability}")
		cv2.imshow('frame',frame)
	except Exception as e:
		print(f"An error occurred: {e}")
		cv2.imshow('frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()