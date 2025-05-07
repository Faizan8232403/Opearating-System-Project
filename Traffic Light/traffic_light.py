import cv2
import torch
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.classes = [2, 3, 5, 7]  


cap = cv2.VideoCapture(r'D:\VS\Traffic Light\traffic.mp4')


ROI_X_START = 35
ret, frame = cap.read()
height, width = frame.shape[:2]
ROI_X_END = width // 3
ROI_Y_START = height // 3
ROI_Y_END = height

speed_factor = 0.7  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    car_count = 0

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

       
        if ROI_X_START <= cx <= ROI_X_END and ROI_Y_START <= cy <= ROI_Y_END:
            car_count += 1
           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

   
    if car_count > 0:
        signal_color = "RED"
        color = (0, 0, 255)  # Red
        red_duration = min(10, 5 + car_count * 0.5) 
    else:
        signal_color = "GREEN"
        color = (0, 255, 0)  
        red_duration = 0  

    
    countdown_time = int(red_duration)

    
    cv2.rectangle(frame, (ROI_X_START, ROI_Y_START), (ROI_X_END, ROI_Y_END), (255, 255, 0), 2)

    
    cv2.putText(frame, f"Signal: {signal_color}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Cars Detected: {car_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

   
    if red_duration > 0:
        cv2.putText(frame, f"Red Light Time: {countdown_time}s", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    cv2.imshow("Traffic Light Simulator", frame)

    
    key = cv2.waitKey(int(1000 * speed_factor)) 

    
    if key == 27:
        break

    
    if red_duration > 0:
        red_duration -= 0.1 

cap.release()
cv2.destroyAllWindows()
