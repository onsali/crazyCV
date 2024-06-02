import torch
import cv2
import numpy as np
import requests
import random

#load yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#class labels
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
           'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
           'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'shirt', 'pants', 'dress',
           'hat', 'tie', 'suit', 'coat', 'sneakers', 'boots', 'gloves', 'scarf', 'sunglasses', 'handkerchief',
           'pen', 'pencil', 'notebook', 'highlighter', 'stapler', 'glue stick', 'tape', 'ruler', 'scissors',
           'lighter']

#init webcam
cap = cv2.VideoCapture(0)

#adjust screen res
screen_width, screen_height = 2048, 1152 

#fakeperson class for fake profile information from namefake API
class FakePerson:
    def __init__(self):
        self.name = ""
        self.company = ""
        self.email = ""
        self.password = ""
        self.mood = ""

    def fetch_fake_profile(self):
        url = 'https://api.namefake.com/'
        response = requests.get(url)
        data = response.json()
        self.name = data.get('name')
        self.company = data.get('company')
        self.email = f"{data.get('email_u')}@{data.get('email_d')}"
        self.password = data.get('password')
        self.mood = random.choice(["happy", "angry", "neutral", "laughing", "crying"])

#dict to map detected people with profile
fake_persons = {}

#random price gen
prices = {class_name: np.random.randint(1, 100) for class_name in classes}

#profile table fn
def draw_table(image, x, y, data, font_size, font_thickness, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255)):
    max_width = max(cv2.getTextSize(col, font, font_size, font_thickness)[0][0] for col in data[0])
    row_height = cv2.getTextSize(data[0][0], font, font_size, font_thickness)[0][1] + 5
    
    for i, row in enumerate(data):
        y_offset = y + i * row_height
        for j, col in enumerate(row):
            cv2.putText(image, col, (x + j * max_width, y_offset), font, font_size, color, font_thickness)

#fn to display glitches
def flash_popup(frame_resized, text, font_size=1.0):
    global dialogues_duration, dialogues_move_times, dialogues_switch_time, prev_dialogue_switch_time, current_dialogue_duration, dialogue_counter, dialogue_x, dialogue_y, dialogue_speed_x, dialogue_speed_y
    
    #timing
    current_time = cv2.getTickCount()
    elapsed_time = (current_time - prev_dialogue_switch_time) / cv2.getTickFrequency()
    
    if elapsed_time > dialogues_switch_time:
        if current_dialogue_duration < 2 * dialogues_duration:
            #text size
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
            
            #update dialogue position
            dialogue_x += dialogue_speed_x
            dialogue_y += dialogue_speed_y
            
            #check boundary
            if dialogue_x < 0 or dialogue_x + text_size[0] > screen_width:
                dialogue_speed_x *= -1
                dialogue_x = max(min(dialogue_x, screen_width - text_size[0]), 0) 
            if dialogue_y < text_size[1] or dialogue_y > screen_height:
                dialogue_speed_y *= -1
                dialogue_y = max(min(dialogue_y, screen_height - 10), text_size[1]) 
            
            #black background
            cv2.rectangle(frame_resized, (dialogue_x - 10, dialogue_y - text_size[1] - 10), (dialogue_x + text_size[0] + 10, dialogue_y + 10), (0, 0, 0), cv2.FILLED)
            
            #test display
            cv2.putText(frame_resized, text, (dialogue_x, dialogue_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
            
            current_dialogue_duration += elapsed_time
        else:
            current_dialogue_duration = 0
            dialogue_counter += 1
            if dialogue_counter >= dialogues_move_times:
                prev_dialogue_switch_time = cv2.getTickCount()
                dialogue_counter = 0

#dialogue switching
dialogues_switch_time = 2  #time interval (in seconds) for switching dialogues
dialogues_duration = 12  #duration (in seconds) for displaying each dialogue 
dialogues_move_times = 2  #number of times the dialogue should move through the screen
prev_dialogue_switch_time = cv2.getTickCount()
current_dialogue_duration = 0
dialogue_counter = 0

#dialogue position and movement
dialogue_x = random.randint(50, screen_width - 150)
dialogue_y = random.randint(100, screen_height - 50)
dialogue_speed_x = random.choice([-1, 1]) 
dialogue_speed_y = random.choice([-1, 1]) 

#list of random flashing dialogues
flash_dialogues = [
    "Have you taken your mandatory dose of electronic supplements today, Citizen 4372?",
    "Upgrade your neural interface for enhanced cognitive abilities! Act now and receive a 10% discount!",
    "Have you changed your passwords, Citizen 4372?",
    "Attention, Citizen 4372! Your biometric data indicates low serotonin levels. How about a refreshing dose of our latest happiness-inducing beverage?",
    "Concerned about privacy? Don't worry, Citizen 4372. Our cyborg implants come with state-of-the-art encryption, ensuring your data is safe with us."
]

#time variables for alternating display
start_time = cv2.getTickCount()
toggle_time = 0
display_person_info = True
dialogue_index = 0

#main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    #convert to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)

    #overlay
    for detection in results.xyxy[0]:
        class_id = int(detection[5])
        confidence = detection[4].item()
        x1, y1, x2, y2 = list(map(int, detection[:4]))
        if confidence > 0.5:
            #draw bounding box
            if classes[class_id] == 'person':
                if class_id not in fake_persons:
                    fake_person = FakePerson()
                    fake_person.fetch_fake_profile()
                    fake_persons[class_id] = fake_person
                else:
                    fake_person = fake_persons[class_id]
                color = (0, 0, 255)  #red for people
                #fake profile info
                person_data = [
                    ["Name:", fake_person.name],
                    ["Company:", fake_person.company],
                    ["Email:", fake_person.email],
                    ["Password:", fake_person.password]
                ]
                draw_table(frame, x1, y1 - 100, person_data, 0.4, 1, cv2.FONT_HERSHEY_SIMPLEX, color)
            else:
                color = (0, 255, 0)  #green for objects
                #add price tag for objects (except people) with green bounding box
                cv2.putText(frame, f"Price: ${prices[classes[class_id]]}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = classes[class_id]
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    #fullscreen
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    #check alternating
    if display_person_info:
        if dialogue_index < len(flash_dialogues):
            flash_popup(frame_resized, flash_dialogues[dialogue_index], font_size=0.8)
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - toggle_time) / cv2.getTickFrequency()
            if elapsed_time > 5:
                dialogue_index += 1
                toggle_time = cv2.getTickCount()
        else:
            #display crazy text
            crazy_text = ''.join(random.sample(flash_dialogues[0], len(flash_dialogues[0])))
            flash_popup(frame_resized, crazy_text, font_size=0.8)
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - toggle_time) / cv2.getTickFrequency()
            if elapsed_time > 5:
                dialogue_index = 0
                toggle_time = cv2.getTickCount()

    cv2.namedWindow("CrazyCV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CrazyCV", 2048, 1152) 
    cv2.imshow("CrazyCV", frame_resized)

    #exit on key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
#release resources
cap.release()
cv2.destroyAllWindows()

