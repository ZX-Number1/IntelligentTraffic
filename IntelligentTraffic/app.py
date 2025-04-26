from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import threading
import collections
import pyttsx3
import pytesseract
import easyocr
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8s.pt")
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()
engine.setProperty('volume', 1)

speaking = False
speaking_lock = threading.Lock()

def synchronized(func):
    def wrapper(*args, **kwargs):
        with speaking_lock:
            return func(*args, **kwargs)
    return wrapper

@synchronized
def text_to_speech(text, volume=1):
    global speaking
    if speaking:
        return
    speaking = True
    engine.setProperty('volume', volume)
    engine.say(text)
    engine.runAndWait()
    speaking = False

def is_daytime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 80

def detect_traffic_light_color(roi, is_day):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if is_day:
        red1 = (np.array([0, 100, 100]), np.array([10, 255, 255]))
        red2 = (np.array([170, 100, 100]), np.array([180, 255, 255]))
        yellow = (np.array([15, 120, 120]), np.array([35, 255, 255]))
        green = (np.array([35, 100, 100]), np.array([85, 255, 255]))
    else:
        red1 = (np.array([0, 50, 50]), np.array([10, 255, 255]))
        red2 = (np.array([170, 50, 50]), np.array([180, 255, 255]))
        yellow = (np.array([15, 50, 50]), np.array([35, 255, 255]))
        green = (np.array([30, 50, 50]), np.array([120, 255, 255]))

    red_mask = cv2.inRange(hsv, *red1) | cv2.inRange(hsv, *red2)
    yellow_mask = cv2.inRange(hsv, *yellow)
    green_mask = cv2.inRange(hsv, *green)

    areas = [cv2.countNonZero(m) for m in [red_mask, yellow_mask, green_mask]]
    roi_size = roi.shape[0] * roi.shape[1]
    areas = [a if a/roi_size > 0.05 else 0 for a in areas]

    if max(areas) == 0:
        return "Unknown"
    return ["Red", "Yellow", "Green"][np.argmax(areas)]

def extract_traffic_light_number(roi, is_day):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    yellow_part = cv2.bitwise_and(roi, roi, mask=mask)
    gray = cv2.cvtColor(yellow_part, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if is_day:
        result = reader.readtext(binary, allowlist="0123456789", detail=0)
    else:
        result = pytesseract.image_to_string(binary, config="--psm 6 digits")
    text = ''.join(filter(str.isdigit, ''.join(result)))
    return text if text else "N/A"

def determine_side(roi, is_day):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_range = (np.array([35, 100, 100]), np.array([85, 255, 255])) if is_day else (np.array([35, 50, 50]), np.array([85, 255, 255]))
    red_range = (np.array([0, 100, 100]), np.array([10, 255, 255])) if is_day else (np.array([0, 50, 50]), np.array([10, 255, 255]))

    green_mask = cv2.inRange(hsv, *green_range)
    red_mask = cv2.inRange(hsv, *red_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(green_mask | red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.3 < w / float(h) < 0.6:
            center_x = x + w // 2
            if center_x < roi.shape[1] // 2:
                return "Left"
            else:
                return "Right"
    return "Straight"

last_traffic_color = "Unknown"
last_truck_alert = False
last_countdown_alert = False
result_queue = collections.deque(maxlen=5)

@socketio.on("stream_frame")
def handle_frame(data):
    global last_traffic_color, last_truck_alert, last_countdown_alert
    try:
        image_data = base64.b64decode(data["image"])
        frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        is_day = is_daytime(frame)
        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)
                roi = frame[y1:y2, x1:x2]

                if conf > 0.3:
                    if cls_id == 7:
                        if not last_truck_alert:
                            text_to_speech("小心大型車")
                            last_truck_alert = True
                        detections.append("🚛 偵測到卡車")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    elif cls_id in [9, 10, 11]:
                        color = detect_traffic_light_color(roi, is_day)
                        countdown = extract_traffic_light_number(roi, is_day)
                        side = determine_side(roi, is_day)

                        if color != last_traffic_color:
                            if color == "Red" or color == "Yellow":
                                text_to_speech("不可通行")
                            elif color == "Green":
                                msg = "可通行，注意大型車" if last_truck_alert else "可通行"
                                text_to_speech(msg)
                            last_traffic_color = color

                        if countdown.isdigit():
                            val = int(countdown)
                            if val < 20 and not last_countdown_alert:
                                text_to_speech("請注意現在秒數低於20秒")
                                last_countdown_alert = True
                            elif val >= 30:
                                last_countdown_alert = False

                        detections.append(f"🚦 紅綠燈: {color}（{side}）")
                        cv2.putText(frame, f"{color} {side} {countdown}s", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frame = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("response_frame", {"image": encoded_frame, "detections": detections})

    except Exception as e:
        print(f"❌ 錯誤: {e}")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    
