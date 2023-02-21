from fer import FER
import cv2


# Use the MTCNN for emotion detection
detector = FER(mtcnn=True)

# Read the input
img_emot = cv2.imread("input_img.jpg")

#Analyize emotions from detected face 
emot_analyze = detector.detect_emotions(img_emot) 
#print(emot_analyze)

#From the dictonary output we see two keys: 1.) "box", 2.) "emotions"
#We can use this to display the result along with the image as follows:
box_cod = emot_analyze[0]["box"]
emotions = emot_analyze[0]["emotions"]

#Draw a circle around the face
#center_coordinates = (box_cod[0] + box_cod[2])//2, (box_cod[1] + box_cod[3])//2
#radius =  int(box_cod[3] / 2 )#box_cod[2]//2 
#Draw a rectangle around the face
cv2.rectangle(img_emot, (box_cod[0], box_cod[1]), (box_cod[0]+box_cod[2], box_cod[1]+box_cod[3]), (155,100,0), 2)

emotion_type, score = detector.top_emotion(img_emot)

#Label the emotions and score for each of them
for index, (emotion_type,score) in enumerate(emotions.items()):
    color = (211, 211,211) if score < 0.01 else (255, 0, 0)
    emot_score = "{}: {}".format(emotion_type, "{:.2f}".format(score))
    cv2.putText(img_emot,emot_score, (box_cod[0], box_cod[1] + box_cod[3] + 30 + index * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)

#Write the output to a new file
cv2.imwrite("emotion.jpg", img_emot)
