import cv2


cap = cv2.VideoCapture("Download.mp4")

while True:
    ret, frame = cap.read()
    if ret:


        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break