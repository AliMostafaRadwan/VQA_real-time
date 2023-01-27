from transformers import ViltProcessor, ViltForQuestionAnswering
import cv2
import threading


# disable warnings
import warnings
warnings.filterwarnings("ignore")


user_input = ""
ans = ""
frame = None

# Create a function to take user input
def get_user_input():
    global user_input
    while True:
        user_input = input("Enter your command: ")
        print("You entered: ", user_input)
        if user_input == 'q':
            break

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


# Open a connection to the webcam
cap = cv2.VideoCapture(1)

# Create a thread to take user input
input_thread = threading.Thread(target=get_user_input)
input_thread.start()



def capture_frame():
    while True:
        # Read a frame from the webcam
        global frame
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in the window
        
        cv2.putText(frame, "Question: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, user_input, (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.putText(frame, "Answer: ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ans, (110, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Webcam Feed", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Check if the user pressed the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_question():
    global ans
    while True:
        if user_input.strip() != "":
            
            encoding = processor(frame, user_input, return_tensors="pt")
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            ans = model.config.id2label[idx]
            # print("Predicted answer:", ans)




frame_thread = threading.Thread(target=capture_frame)
question_thread = threading.Thread(target=process_question)
frame_thread.start()
question_thread.start()
