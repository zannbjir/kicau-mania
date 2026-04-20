import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

MODEL_PATH = "hand_landmarker.task"
VIDEO_PATH = "kicau.mp4"
CAM_W, CAM_H = 640, 360 # Resolusi kecil biar enteng/gak lag

# Inisialisasi Video Kucing
cap_video = cv2.VideoCapture(VIDEO_PATH)

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

gesture_history = []

print("=== UDAH JALAN WOI! ===")
print("Lakukan pose jari di mulut untuk buka tab video kucing.")
print("Tekan 'x' pada keyboard untuk berhenti.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    # AI memproses gambar yang lebih kecil biar FPS tinggi (lancar)
    small_frame = cv2.resize(frame, (320, 180))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
    
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    detected_now = "None"
    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            i8 = landmarks[8] # Ujung telunjuk
            # Cek apakah jari tengah dan manis menekuk
            is_closed = (landmarks[12].y > landmarks[10].y and 
                         landmarks[16].y > landmarks[14].y)
            
            print(f"Detecting... Index Y: {i8.y:.2f} | Others Closed: {is_closed}")

            # Syarat: Telunjuk di area mulut & jari lain nutup
            if i8.y < 0.6 and abs(i8.x - 0.5) < 0.2 and is_closed:
                detected_now = "Mikir"

    gesture_history.append(detected_now)
    if len(gesture_history) > 3: gesture_history.pop(0)
    stable_gesture = gesture_history[0] if len(set(gesture_history)) == 1 else "None"

    if stable_gesture == "Mikir":
        rv, vf = cap_video.read()
        if not rv: # Loop video kalau habis
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            rv, vf = cap_video.read()
        
        if rv:
            # Tampilkan di TAB BARU bernama "Kicau Mania"
            cv2.imshow("Kicau Mania", vf)
    else:
        # Tutup tab kalau gak lagi pose
        # Pengecekan agar tidak error saat mencoba menutup jendela yang sudah tutup
        try:
            if cv2.getWindowProperty("Kicau Mania", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Kicau Mania")
        except:
            pass
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow("Kicau Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'): break

cap.release()
cap_video.release()
cv2.destroyAllWindows()