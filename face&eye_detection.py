import cv2



# Mention absolute path if you get an assertion error (not mentioned here) for the haarcascade classifiers

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Load the Eye Classifir
# haarcascade_eye_tree_eyeglasses.xml -> Can detect eyes even with spectacles more accurately than haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')



# Initialize the video capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the frame width and height
cap.set(3, 640)


while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    #img = cv2.flip(frame,1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(gray)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame_gray)


    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 2)

        #Extract ROI of face to make it easier to search for eyes in each face detected
        roi_face = frame_gray[y: y+h, x: x+w]
        eye = eye_cascade.detectMultiScale(roi_face)

        for (x2,y2,w2,h2) in eye:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25)) 
            cv2.circle(frame, eye_center, radius, (255,0,0), 2)
        

    # Show the frame
    cv2.imshow('Frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()

