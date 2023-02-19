import cv2


# Load the face cascade classifier (Use absolute path if it gives an Assertion error)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Check if camera is functional
if not cap.isOpened():
    print("Unable to open camera")
    exit()

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    
    rows, cols, _ = frame.shape
    gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    faces = face_cascade.detectMultiScale(gray_roi, 1.3, 5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:

         # Get the region of interest for the eyes
        roi_gray = gray_roi[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the region of interest
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)

        # Loop through the detected eyes
        for (ex, ey, ew, eh) in eyes:
   
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.line(roi_color, ((ex) + int(ew/2), 0), (ex + int(ew/2), rows), (0, 255, 0), 2)
            cv2.line(roi_color, (0, (ey) + int(eh/2)), (cols, (ey) + int(eh/2)), (0, 255, 0), 2)
            break
    cv2.imshow("Roi", frame)

    #press 'q' key to exit()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
