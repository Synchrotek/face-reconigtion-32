import cv2
import os
import dlib
import face_recognition
import threading

# Load the known faces and their names from the "faces" folder
known_faces = []
known_names = []

# Replace the paths with your folder and image extensions as needed
face_images_folder = "faces/"
image_extensions = (".jpg", ".jpeg", ".png")

for file in os.listdir(face_images_folder):
    if file.lower().endswith(image_extensions):
        image = face_recognition.load_image_file(os.path.join(face_images_folder, file))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
        known_names.append(os.path.splitext(file)[0])

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()


# Function for face recognition
def recognize_faces():
    global video_capture
    global known_faces
    global known_names
    global detector
    tracked_faces = {}
    fps_reset_interval = 30  # Number of frames to reset FPS after detecting a face
    fps_reset_counter = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Find face locations in the current frame using dlib's face detector
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = detector(gray_frame)

        # Loop through each face found in the frame
        for face_location in face_locations:
            top, right, bottom, left = (
                face_location.top(),
                face_location.right(),
                face_location.bottom(),
                face_location.left(),
            )
            face_image = frame[top:bottom, left:right]

            # Compute face encoding for the detected face
            face_encoding = face_recognition.face_encodings(face_image)

            # Check if the face matches any known face
            name = "Unknown"  # Default name is Unknown if no match is found
            if len(face_encoding) > 0:
                matches = face_recognition.compare_faces(known_faces, face_encoding[0])
                if True in matches:
                    name = known_names[matches.index(True)]

            # Draw a rectangle around the face and write the name on it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                1,
            )

            # Track the face using a simple dictionary
            tracked_faces[name] = (left, top, right, bottom)

            # Reset the FPS counter if a face is detected
            fps_reset_counter = fps_reset_interval

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Decrement the FPS reset counter
        fps_reset_counter -= 1

        # Reset the FPS to 60 after the specified interval if no face is detected
        if fps_reset_counter <= 0:
            fps_reset_counter = 0

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Start the face recognition thread
face_recognition_thread = threading.Thread(target=recognize_faces)
face_recognition_thread.start()

# Wait for the face recognition thread to finish
face_recognition_thread.join()

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
