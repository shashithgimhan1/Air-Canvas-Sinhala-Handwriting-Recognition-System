import os
import math
import cv2
import numpy as np
import tf_keras as keras
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

label_dict = {0: 'අ', 1: 'එ', 2: 'ඉ', 3: 'උ', 4: 'ස', 5: 'ර', 6: 'ව'}

def decode_prediction(class_idx):
    return label_dict.get(class_idx, "")

try:
    model = keras.models.load_model('drawing_model.keras')
except Exception as e:
    print("Error loading model. Did you run train_model.py first?")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

canvas = None
ai_canvas = None 
px, py = 0, 0
prediction = ""
draw_color = (255, 255, 255) 
ink_transparency = 1.0 

letter_gap = 2            
smoothing_factor = 0.25    

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        ai_canvas = np.ones((h, w), dtype=np.uint8) * 255 

    cv2.rectangle(frame, (160, 10), (240, 60), (0, 0, 255), -1)      
    cv2.rectangle(frame, (260, 10), (340, 60), (0, 255, 0), -1)      
    cv2.rectangle(frame, (360, 10), (440, 60), (255, 0, 0), -1)      
    cv2.rectangle(frame, (460, 10), (540, 60), (255, 255, 255), -1)  
    cv2.rectangle(frame, (460, 10), (540, 60), (0, 0, 0), 2)         

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            raw_cx, raw_cy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            distance = math.hypot(raw_cx - tx, raw_cy - ty)
            
            cv2.circle(frame, (raw_cx, raw_cy), 10, draw_color, cv2.FILLED) 
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), cv2.FILLED)

            if distance > 40: 
                if px == 0 and py == 0:
                    px, py = raw_cx, raw_cy
                
                cx = int((raw_cx * smoothing_factor) + (px * (1.0 - smoothing_factor)))
                cy = int((raw_cy * smoothing_factor) + (py * (1.0 - smoothing_factor)))
                
                if cy <= 60: 
                   
                    if 160 <= raw_cx <= 240:    draw_color = (0, 0, 255)     
                    elif 260 <= raw_cx <= 340:  draw_color = (0, 255, 0)     
                    elif 360 <= raw_cx <= 440:  draw_color = (255, 0, 0)     
                    elif 460 <= raw_cx <= 540:  draw_color = (255, 255, 255) 
                    px, py = 0, 0 # Lifts pen
                elif cy >= h - 60 or cx <= 30 or cx >= w - 30:
                   
                    px, py = 0, 0 # Lifts pen to stop jitter
                else:
                   
                    cv2.line(canvas, (px, py), (cx, cy), draw_color, 12)
                    cv2.line(ai_canvas, (px, py), (cx, cy), 0, 12)
                    px, py = cx, cy
            else:
                px, py = 0, 0
                cv2.circle(frame, ((raw_cx+tx)//2, (raw_cy+ty)//2), 15, (0, 255, 0), cv2.FILLED)
    else:
        px, py = 0, 0 

    overlay = frame.copy()
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ink_mask = canvas_gray > 0 
    
    overlay[ink_mask] = canvas[ink_mask]
    blended_frame = cv2.addWeighted(overlay, ink_transparency, frame, 1 - ink_transparency, 0)

    ai_canvas_inverted = cv2.bitwise_not(ai_canvas)
    contours, _ = cv2.findContours(ai_canvas_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    valid_boxes = [b for b in bounding_boxes if b[2] > 10 and b[3] > 10]
    valid_boxes.sort(key=lambda b: b[0])
    
    merged_boxes = []
    for box in valid_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            last_box = merged_boxes[-1]
            x, y, bw, bh = box
            lx, ly, lbw, lbh = last_box
            
            gap = x - (lx + lbw)
            
            if gap < letter_gap:
                new_x = min(x, lx)
                new_y = min(y, ly)
                new_w = max(x + bw, lx + lbw) - new_x
                new_h = max(y + bh, ly + lbh) - new_y
                merged_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(box)
                
    for (x, y, bw, bh) in merged_boxes:
        cv2.rectangle(blended_frame, (x - 10, y - 10), (x + bw + 10, y + bh + 10), (0, 255, 255), 2)

    cv2.putText(blended_frame, "Controls: ENTER (Predict) | 'c' (Clear) | 'q' (Quit)", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img_pil = Image.fromarray(cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("iskpota.ttf", 50) 
    except IOError:
        try:
            font = ImageFont.truetype("Nirmala.ttf", 50) 
        except IOError:
            font = ImageFont.load_default()

    rgb_color = (draw_color[2], draw_color[1], draw_color[0])
    draw.text((20, 80), f"Prediction: {prediction}", font=font, fill=rgb_color)
    blended_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("My Drawing", blended_frame)

    key = cv2.waitKey(1) & 0xFF
        
    if key == ord('c'): 
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        ai_canvas = np.ones((h, w), dtype=np.uint8) * 255 
        prediction = ""

    elif key == 13: 
        full_word = ""
        letter_count = 0
        
        for (x, y, bw, bh) in merged_boxes:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            cropped_image = ai_canvas[y1:y2, x1:x2]
            
            if cropped_image.size > 0:
                ch, cw = cropped_image.shape
                
                scale = 20.0 / max(ch, cw)
                new_w, new_h = int(cw * scale), int(ch * scale)
                
                if new_w <= 0 or new_h <= 0:
                    continue
                    
                resized_crop = cv2.resize(cropped_image, (new_w, new_h))
                
                _, resized_crop = cv2.threshold(resized_crop, 150, 255, cv2.THRESH_BINARY)
                kernel = np.ones((2, 2), np.uint8)
                resized_crop = cv2.erode(resized_crop, kernel, iterations=1)

                final_img = np.ones((28, 28), dtype=np.uint8) * 255
                x_offset = (28 - new_w) // 2
                y_offset = (28 - new_h) // 2
                final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop
                
                normalized = final_img / 255.0
                reshaped = np.reshape(normalized, (1, 28, 28, 1))
                
                pred_array = model.predict(reshaped, verbose=0)
                class_idx = np.argmax(pred_array[0])
                full_word += decode_prediction(class_idx)
        
        prediction = full_word
        
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        ai_canvas = np.ones((h, w), dtype=np.uint8) * 255

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()