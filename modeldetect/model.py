import cv2
from ultralytics import YOLO
import numpy as np
import math

# Load fast YOLO model
model = YOLO("yolov8n.pt")

video = cv2.VideoCapture(
    r"D:\Rider_Safety_System\modeldetect\Screen Recording 2026-05-26 230129.mp4"
)

if not video.isOpened():
    print("Video not found")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

# Store previous positions
previous_centers = {}

frame_count = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1

    # Skip some frames for speed
    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640,360))

    h,w = frame.shape[:2]

    # Green road region
    overlay = frame.copy()

    road_points=np.array([
        [0,h],
        [w,h],
        [int(w*0.7),int(h*0.6)],
        [int(w*0.3),int(h*0.6)]
    ])

    cv2.fillPoly(
        overlay,
        [road_points],
        (0,255,0)
    )

    frame=cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0
    )

    results=model.track(
        frame,
        persist=True,
        classes=[2,3,5,7], # car,motorbike,bus,truck
        verbose=False
    )

    for r in results:

        if r.boxes.id is None:
            continue

        boxes=r.boxes.xyxy.cpu().numpy()
        ids=r.boxes.id.cpu().numpy()

        for box,track_id in zip(boxes,ids):

            x1,y1,x2,y2=map(int,box)

            center=(
                (x1+x2)//2,
                (y1+y2)//2
            )

            speed=0

            if track_id in previous_centers:

                prev_center=previous_centers[track_id]

                # Distance moved in pixels
                distance=math.sqrt(
                    (center[0]-prev_center[0])**2+
                    (center[1]-prev_center[1])**2
                )

                # Approximate conversion
                meters=distance*0.05

                # Speed calculation
                speed=(meters*fps)*3.6

            previous_centers[track_id]=center

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                f"ID:{int(track_id)} {speed:.1f} km/h",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

    cv2.imshow(
        "Road + Vehicle Speed Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()