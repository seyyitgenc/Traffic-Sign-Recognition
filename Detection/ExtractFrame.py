import cv2
import csv
import os

def returnRedness(img):
	yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	return v

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
	# img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	sign = img[y:(y+h) , x:(x+w)]
	return img, sign

def preprocessingImageToClassifier(image=None,imageSize=28,mu=89.77428691773054,std=70.85156431910688):
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image,(imageSize,imageSize))
	image = cv2.equalizeHist(image)
	image = image/255
	image = image.reshape(1,imageSize,imageSize,1)
	return image


# Open the video file
cap = cv2.VideoCapture('C:/projects_ai/Traffic-Sign-Recognition/DatasetVideos/turn_left.mp4')

# Check if video opened successfully
if not cap.isOpened(): 
	print("Error opening video file")

# Specify the directory where the images are saved
image_dir = 'C:/Datasets/ituro_traffic_signs/1/'

# Read until video is completed
frame_count = 0
while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret:
		redness = returnRedness(frame)
		thresh = threshold(redness) 	
		contours = findContour(thresh)
		big = findBiggestContour(contours)
		if cv2.contourArea(big) > 3000:
			img,sign = boundaryBox(frame,big)
			cv2.imwrite(f'{image_dir}frame{frame_count}.png', sign)
			frame_count += 1
	else: 
		break

# When everything done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()

# Get a list of all image file names in the directory
image_files = os.listdir(image_dir)

# Open a new CSV file in write mode
with open('image_data.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	# Write the header row
	writer.writerow(["FileName", "Label"])
	# Write a row for each image file
	for image_file in image_files:
		# Use the file name as the label for now
		label = "turn_left"
		writer.writerow([image_file, label])