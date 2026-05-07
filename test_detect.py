import cv2, subprocess, sys, os
sys.path.insert(0, '.')
from blur_plates import load_models, apply_blur

subprocess.run([
    'ffmpeg', '-y', '-ss', '89', '-i', '../final.mov',
    '-frames:v', '1', '-f', 'image2', 'test_frame.png'
], capture_output=True)

frame = cv2.imread('test_frame.png')
h, w = frame.shape[:2]
print(f'Frame: {w}x{h}')

yolo, cascades, device = load_models()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# YOLO vehicle detection on full frame
results = yolo(frame, conf=0.3, verbose=False)
vehicle_boxes = []
VEHICLE_CLASSES = {2, 3, 5, 7}
for r in results:
    if r.boxes is None: continue
    for box in r.boxes:
        if int(box.cls[0]) in VEHICLE_CLASSES:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            vehicle_boxes.append((x1,y1,x2,y2))
            print(f'  Vehicle detected: ({x1},{y1})-({x2},{y2})')

print(f'Total vehicles: {len(vehicle_boxes)}')

# Cascade on full-res frame
plate_rects = []
for casc in cascades:
    dets = casc.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(60,20), maxSize=(500,150))
    for (px,py,pw,ph) in (dets if len(dets) else []):
        print(f'  Cascade hit: ({px},{py}) {pw}x{ph}')
        plate_rects.append((px, py, px+pw, py+ph))

# Also search inside vehicle boxes
for (vx1,vy1,vx2,vy2) in vehicle_boxes:
    mid_y = vy1 + int((vy2-vy1)*0.4)
    roi = gray[max(0,mid_y):min(h,vy2+20), max(0,vx1-10):min(w,vx2+10)]
    for casc in cascades:
        dets = casc.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=3, minSize=(60,20), maxSize=(500,150))
        for (px,py,pw,ph) in (dets if len(dets) else []):
            print(f'  Plate in vehicle: ({vx1+px},{mid_y+py}) {pw}x{ph}')
            plate_rects.append((vx1+px, mid_y+py, vx1+px+pw, mid_y+py+ph))

print(f'Total plate regions: {len(plate_rects)}')

if plate_rects:
    out = apply_blur(frame.copy(), plate_rects)
    cv2.imwrite('test_frame_blurred.png', out)
    print('Saved test_frame_blurred.png')
else:
    print('NO PLATES DETECTED — check test_frame.png manually')

