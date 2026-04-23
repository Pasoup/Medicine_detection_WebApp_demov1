import cv2
import os
import time

# --- config ---
SAVE_PATH = "medicine_dataset"
MEDICINE_NAME = "qr_med" 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

count = 0
print(f"Ready to capture. Saving to: {SAVE_PATH}")
print("Press 'SPACE' to take a photo | Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    preview = frame.copy()
    cv2.putText(preview, f"Count: {count} | Label: {MEDICINE_NAME}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Capture Rig - Press Space", preview)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        filename = f"{MEDICINE_NAME}_{int(time.time())}.jpg"
        file_path = os.path.join(SAVE_PATH, filename)
        cv2.imwrite(file_path, frame) 
        count += 1
        print(f"Saved: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()