import cv2
from fer import FER  # Facial Expression Recognition library
import matplotlib.pyplot as plt

# Load the pre-trained FER model
detector = FER()

# Access the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Use FER to detect emotions in the frame
    result = detector.detect_emotions(frame)
    
    # Draw bounding box and label on the face for each detected emotion
    if result:
        for face in result:
            # Get the bounding box for the face
            x, y, w, h = face['box']
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Get the dominant emotion
            emotion = max(face['emotions'], key=face['emotions'].get)
            
            # Put the emotion text above the rectangle
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video_capture.release()
cv2.destroyAllWindows()