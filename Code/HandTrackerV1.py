import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up Hand Tracking model
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        # Flip the image horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands and landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmarks for fingertips and wrist
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Convert landmarks to pixel coordinates
                h, w, _ = frame.shape
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Calculate distances between key points (index-tip to wrist, thumb-tip to wrist)
                index_to_wrist = abs(index_y - wrist_y)
                thumb_to_wrist = abs(thumb_y - wrist_y)

                # Determine if the hand is open or closed based on finger positions
                if index_to_wrist > thumb_to_wrist + 20:
                    cv2.putText(frame, "Gesture: Open Hand", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Gesture: Closed Hand", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw a circle at the index finger tip
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
