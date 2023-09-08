import subprocess
import cv2
import numpy as np
import tensorflow as tf
import imutils
from solver import *
classes = np.arange(0, 10)

model = tf.keras.models.load_model('model-OCR.h5')
#print(model.summary())
input_size = 48
direction =-1
row=9
col=0


num_rows = 9
num_cols = 9

cap = cv2.VideoCapture(0)

# Reading a single frame from the camera

ret, frame = cap.read()

# Release the capture

cap.release()

# Save the captured frame as "board.jpg" on the desktop

cv2.imwrite('/home/obaida/Desktop/MohammadProject/camera.jpg', frame)

# Destroy any remaining windows
cv2.destroyAllWindows()


# Convert the 1D array to a 2D matrix (3x4 grid in this example)


# Define the command to run the other Python script
port= "/dev/ttyUSB0"
sendFile="gcodes/send.py"
start = ["python", sendFile, "-p",port, "-f", "gcodes/start.gcode", "-r", "1", "-v", "1"]
up = ["python", sendFile, "-p",port, "-f", "gcodes/up.gcode", "-r", "1", "-v", "1"]
down = ["python", sendFile, "-p",port, "-f", "gcodes/down.gcode", "-r", "1", "-v", "1"]
left = ["python", sendFile, "-p",port, "-f", "gcodes/left.gcode", "-r", "1", "-v", "1"]
right = ["python", sendFile, "-p",port, "-f", "gcodes/right.gcode", "-r", "1", "-v", "1"]
num1 = ["python", sendFile, "-p",port, "-f", "gcodes/1.gcode", "-r", "1", "-v", "1"]
num2 = ["python", sendFile, "-p",port, "-f", "gcodes/2.gcode", "-r", "1", "-v", "1"]
num3 = ["python", sendFile, "-p",port, "-f", "gcodes/3.gcode", "-r", "1", "-v", "1"]
num4 = ["python", sendFile, "-p",port, "-f", "gcodes/4.gcode", "-r", "1", "-v", "1"]
num5 = ["python", sendFile, "-p",port, "-f", "gcodes/5.gcode", "-r", "1", "-v", "1"]
num6 = ["python", sendFile, "-p",port, "-f", "gcodes/6.gcode", "-r", "1", "-v", "1"]
num7 = ["python", sendFile, "-p",port, "-f", "gcodes/7.gcode", "-r", "1", "-v", "1"]
num8 = ["python", sendFile, "-p",port, "-f", "gcodes/8.gcode", "-r", "1", "-v", "1"]
num9 = ["python", sendFile, "-p",port, "-f", "gcodes/9.gcode", "-r", "1", "-v", "1"]

def move():
     

 
        
 
    global  direction, num_rows, num_cols, row, col,start
    
    two_dim_matrix = np.reshape(flat_solved_board_nums, (num_rows, num_cols))
    
   # subprocess.run(start)
    for count_col in range(9):
        for count_row in range(9):
            row+=direction
            if row==9 or row==-1:
                direction*=-1
                col+=1
            if row==9:
                row=8
            elif row==-1:
                row=0
            if col==9:
                break
        
            send(two_dim_matrix[row][col])
    

def send(a):
    global  direction
    if(a==1):
        subprocess.run(num1)
      
    elif(a==2):
        subprocess.run(num2)
    elif(a==3):
        subprocess.run(num3)
    elif(a==4):
        subprocess.run(num4)    
    elif(a==5):
        subprocess.run(num5)     
    elif(a==6):
        subprocess.run(num6)     
    elif(a==7):
         subprocess.run(num7)     
    elif(a==8):
         subprocess.run(num8) 
    elif(a==9):
         subprocess.run(num9)

        
         
    if(direction==1):
        if(row==8):
             subprocess.run(up)
        else:
              
             subprocess.run(right)
           
    else:
         if(row==0):
             subprocess.run(up)
         else:
             subprocess.run(left)
                            
                            
                                      
             
    

def get_perspective(img, location, height = 900, width = 900):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def get_InvPerspective(img, masked_num, location, height = 900, width = 900):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result





def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg) ### ارجع هون اخفي


    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location


# split the board into 81 individual images
def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            cv2.imshow("Splitted block", box)
            cv2.waitKey(50)
            boxes.append(box)
    cv2.destroyAllWindows()
    return boxes

def displayNumbers(img, numbers, color=(0, 255, 0)):
    """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

# Read image
img = cv2.imread('3.png')

# extract board from input image
board, location = find_board(img)


gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
rois = split_boxes(gray)
rois = np.array(rois).reshape(-1, input_size, input_size, 1)

# get prediction
prediction = model.predict(rois)
# print(prediction)

predicted_numbers = []
# get classes from prediction
for i in prediction: 
    index = (np.argmax(i)) # returns the index of the maximum number of the array
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

 #print(predicted_numbers)

# reshape the list 
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)



# solve the board
try:
    solved_board_nums = get_board(board_num)
    #print(solved_board_nums)
    # create a binary array of the predicted numbers. 0 means unsolved numbers of sudoku and 1 means given number.
    binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
    #print(binArr)
    # get only solved numbers for the solved board
    flat_solved_board_nums = solved_board_nums.flatten()*binArr
    # print(flat_solved_board_nums)
    # create a mask
    mask = np.zeros_like(board)

    # displays solved numbers in the mask in the same position where board numbers are empty
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
   # cv2.imshow("Solved Mask", solved_board_mask)
    inv = get_InvPerspective(img, solved_board_mask, location)
    # cv2.imshow("Inverse Perspective", inv)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
    cv2.imshow("Final result", combined)
    # cv2.waitKey(0)
    

except:
    print("Solution doesn't exist. Model misread digits.")


move()
        
                     
        
        



cv2.imshow("Input image", img)
subtract = 255 - inv
cv2.imshow("Inverse Perspective", subtract)
cv2.imwrite('final.jpg', subtract)
#cv2.imshow("Board", board)
cv2.waitKey(0)
cv2.destroyAllWindows()



