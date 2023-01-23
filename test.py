# import cv2
# import threading

# user_input = ""

# # Create a function to take user input
# def get_user_input():
#     global user_input
#     while True:
#         user_input = input("Enter your command: ")
#         print("You entered: ", user_input)
#         if user_input == 'q':
#             break

# # Create a window to display the webcam feed
# cv2.namedWindow("Webcam Feed")

# # Open a connection to the webcam
# cap = cv2.VideoCapture(1)

# # Create a thread to take user input
# input_thread = threading.Thread(target=get_user_input)
# input_thread.start()

# while True:
#     # Read a frame from the webcam
#     _, frame = cap.read()

#     # Display the frame in the window
#     cv2.putText(frame, "Command: ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#     cv2.putText(frame, user_input, (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#     cv2.imshow("Webcam Feed", frame)

#     # Check if the user pressed the 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and destroy the window
# cap.release()
# cv2.destroyAllWindows()



from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "https://farm6.staticflickr.com/5146/5769430482_61695d5186_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "what color is the shirt?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
