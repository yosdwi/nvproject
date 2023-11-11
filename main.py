from DefineDetect import *
from ttscoba import texttospeech
from keras.models import load_model
from speechtotext import speechtotext

def main_lagi():
    loaded_model = load_model('delapan tiga.h5')

    DATA_PATH = os.path.join('DATASET') 
    actions = os.listdir(DATA_PATH)

    sequence = []
    sentence = []
    threshold = 0.6
    accuracy = 0  # Initialize accuracy to 0

    cap = cv2.VideoCapture(1)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            # Draw landmarks
            draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
                max_prob = np.max(res)
                if max_prob > threshold:
                    predicted_action = actions[np.argmax(res)]
                    accuracy = max_prob
                    #print("Predicted Action:", predicted_action)
                    #print("Accuracy:", accuracy)
                    sentence.append(predicted_action)

            if len(sentence) > 1:
                sentence = sentence[-1:]
          
            display_teks(image, results, sentence, accuracy)
             
            texttospeech(sentence)
        
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
       
while True:
    main_lagi()  
    speechtotext()


