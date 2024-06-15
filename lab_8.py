import cv2
import numpy as np

#Задание 1

image = cv2.imread("images/variant-10.jpg", cv2.IMREAD_GRAYSCALE)

ret, thresholded_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

cv2.imshow("changed image", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Задание 2

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

square_size = 150
x_square = int((frame_width - square_size) / 2)
y_square = int((frame_height - square_size) / 2)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, np.array([0, 0, 0]), np.array([180, 255, 30]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if x_square <= x <= x_square + square_size and y_square <= y <= y_square + square_size:
            frame = cv2.flip(frame, 0)  

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
           break
    else:
       break

cap.release()
cv2.destroyAllWindows()


#Доп_задание

fly_image = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

square_size = 150
x_square = int((frame_width - square_size) / 2)
y_square = int((frame_height - square_size) / 2)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, np.array([0, 0, 0]), np.array([180, 255, 30]))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2

                fly_height, fly_width, _ = fly_image.shape

                fly_x = center_x - fly_width // 2
                fly_y = center_y - fly_height // 2

                if fly_y >= 0 and fly_x >= 0 and fly_y + fly_height <= frame.shape[0] and fly_x + fly_width <= frame.shape[1]:
                    for c in range(3):
                        frame[fly_y:fly_y + fly_height, fly_x:fly_x + fly_width, c] = \
                            fly_image[:, :, c] * (fly_image[:, :, 3] / 255) + \
                            frame[fly_y:fly_y + fly_height, fly_x:fly_x + fly_width, c] * (1 - fly_image[:, :, 3] / 255)
                        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()