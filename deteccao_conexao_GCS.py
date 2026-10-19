from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo26s.pt")

def detectar_alvos(frame):
	results = model(frame, verbose=False, stream_buffer=False, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
	alvos = []
	annotated_frame = results[0].plot()
	for result in results[0].boxes.data:
		cx = int((result[0].item() + result[2].item())/2)
		cy = int((result[1].item() + result[3].item())/2)
		conf = result[4].item()
		cls = result[5].item()
		alvos.append([cx, cy, conf, cls])
		cv2.circle(annotated_frame, center=(cx, cy), radius=5, color=(0, 255, 0), thickness=2)
	cv2.imshow("Yolo Inference", annotated_frame)
	return alvos

def solicitar_telemetria(): #placeholder
	x = 0.0
	y = 0.0
	z = 0.0
	yaw = 0.0
	return [x, y, z, yaw]

connection = mavutil.mavlink_connection("172.17.64.1:14550")
connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (connection.target_system, connection.target_component))

cap = cv2.VideoCapture(0)
dados = []

while cap.isOpened():
	msg = connection.recv_match(blocking=True)
	print(msg)
	success, frame = cap.read()
	if success:
		alvos = detectar_alvos(frame)
		telemetria = solicitar_telemetria()
		dados.append([alvos, telemetria])
		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break
	else:
		break
cap.release()
cv2.destroyAllWindows()

print(dados)