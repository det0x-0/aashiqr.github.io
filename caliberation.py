import cv2
import mediapipe as mp
import numpy as np
import time

class PoseCalibrator:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.7)
        self.reference_pose = None 
        
        # --- NEW: AUTO-CAPTURE VARIABLES ---
        self.last_pose = None
        self.still_start_time = None
        self.hold_duration = 2.0  # Seconds you must stay still
        self.still_threshold = 0.005 # How much movement is allowed

    def get_neutrality_score(self, frame):
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return 0, "NO PERSON DETECTED", (0, 0, 255), False

        current_pose = np.array([[l.x, l.y] for l in results.pose_landmarks.landmark])

        # --- PHASE 1: AUTO-CAPTURE LOGIC ---
        if self.reference_pose is None:
            if self.last_pose is not None:
                # Check how much you moved since the last frame
                movement = np.mean(np.linalg.norm(self.last_pose - current_pose, axis=1))
                
                if movement < self.still_threshold:
                    if self.still_start_time is None:
                        self.still_start_time = time.time()
                    
                    elapsed = time.time() - self.still_start_time
                    remaining = max(0, self.hold_duration - elapsed)
                    
                    if elapsed >= self.hold_duration:
                        self.reference_pose = current_pose
                        return 100, "POSE CAPTURED!", (0, 255, 0), True
                    
                    self.last_pose = current_pose
                    return int((elapsed/self.hold_duration)*100), f"HOLD STILL... {remaining:.1f}s", (0, 255, 255), False
                else:
                    # You moved too much, reset the timer
                    self.still_start_time = None
            
            self.last_pose = current_pose
            return 0, "STAY STILL TO CAPTURE", (255, 255, 255), False

        # --- PHASE 2: MATCHING LOGIC ---
        dist = np.mean(np.linalg.norm(self.reference_pose - current_pose, axis=1))
        score = int(max(0, 100 - (dist * 1000))) 
        
        color = (0, 255, 0) if score > 85 else (0, 165, 255) if score > 65 else (0, 0, 255)
        msg = "MATCH FOUND" if score > 85 else "RETURN TO SAVED POSE"
        return score, msg, color, False

# --- RUNNER ---
cap = cv2.VideoCapture(0)
calibrator = PoseCalibrator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    score, message, color, captured = calibrator.get_neutrality_score(frame)

    # UI
    cv2.putText(frame, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (10, 70), (10 + (int(score)*2), 90), color, -1)
    
    if captured:
        # Visual feedback when capture happens
        cv2.rectangle(frame, (0,0), (640, 480), (0, 255, 0), 20)

    cv2.imshow('Hands-Free Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()