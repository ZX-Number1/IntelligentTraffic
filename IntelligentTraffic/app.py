from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import threading
import pyttsx3
import pytesseract
import easyocr
from ultralytics import YOLO
from collections import deque
import time
from queue import Queue


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8s.pt")
reader = easyocr.Reader(['en'])
engine = pyttsx3.init()
engine.setProperty('volume', 1)

speaking = False
speaking_lock = threading.Lock()
engine = pyttsx3.init()
speech_queue = Queue()

# 語音同步鎖定
def synchronized(func):
    def wrapper(*args, **kwargs):
        with speaking_lock:
            return func(*args, **kwargs)
    return wrapper

def speak_worker():
    while True:
        text, volume = speech_queue.get()
        print(f"🔊 播放語音：{text}")
        try:
            engine.setProperty('volume', volume)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ 語音錯誤：{e}")
        speech_queue.task_done()

def speak_queued(text, volume=1):
    speech_queue.put((text, volume))

# ✅ 啟動語音播放執行緒
threading.Thread(target=speak_worker, daemon=True).start()

# 日夜判斷
def is_daytime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 80

# 紅綠燈顏色判斷
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
    areas = [a if a / roi_size > 0.05 else 0 for a in areas]

    if max(areas) == 0:
        return "Unknown"
    return ["Red", "Yellow", "Green"][np.argmax(areas)]

# 倒數數字辨識
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

last_traffic_color = "Unknown"
last_truck_alert = False
last_countdown_alert = False
result_queue = deque(maxlen=5)

last_speak_time = 0
speak_cooldown = 5  # 秒

last_warn_time = 0
warn_cooldown = 10  # 秒

@socketio.on("stream_frame")
def handle_frame(data):
    global last_traffic_color, last_truck_alert, last_countdown_alert
    global last_speak_time, speak_cooldown, last_warn_time, warn_cooldown
    global result_queue

    print("🛠️ handle_frame 被呼叫")

    try:
        image_data = base64.b64decode(data["image"])
        frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        print(f"✅ 解碼成功，影像尺寸: {frame.shape}")
        is_daytime_scene = is_daytime(frame)
        results = model(frame)
        detections = []

        frame_count = int(time.time() * 1000) % 1000
        current_time = time.time()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls)

                print(f"🔍 偵測類別: {class_id}, 置信度: {conf:.2f}")

                # 卡車偵測
                if conf > 0.3 and class_id == 7:
                    print("🚛 偵測到卡車")
                    label = f"Truck {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if not last_truck_alert:
                        speak_queued("小心大型車", volume=1)
                        last_truck_alert = True
                    detections.append("🚛 偵測到卡車")

                if frame_count % 50 == 0:
                    last_truck_alert = False

                # 紅綠燈辨識
                if conf > 0.3 and class_id in [9, 10, 11]:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        print("🚦 處理紅綠燈 ROI")
                        color = detect_traffic_light_color(roi, is_daytime_scene)
                        print(f"🎨 初步顏色偵測結果: {color}")

                        result_queue.append(color)
                        red_count = result_queue.count("Red")
                        yellow_count = result_queue.count("Yellow")
                        green_count = result_queue.count("Green")

                        if red_count >= 2:
                            color = "Red"
                        elif yellow_count >= 2:
                            color = "Yellow"
                        elif green_count >= 2:
                            color = "Green"
                        else:
                            color = "Unknown"
                        print(f"📊 平滑後顏色: {color}")

                        box_color = (0, 255, 0) if color == "Green" else \
                                    (0, 255, 255) if color == "Yellow" else \
                                    (0, 0, 255) if color == "Red" else \
                                    (255, 255, 255)
                        label = f"{color} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                        countdown_number = extract_traffic_light_number(roi, is_daytime_scene)
                        print(f"⏱️ 倒數時間辨識結果: {countdown_number}")

                        # ✅ 播報條件分兩種：變色 or 超過冷卻時間
                        if color != "Unknown":
                            if color != last_traffic_color:
                                print("🟢 顏色變化 → 播報")
                                if color == "Red":
                                    speak_queued("不可通行", volume=1)
                                elif color == "Yellow":
                                    speak_queued("不可通行", volume=1)
                                elif color == "Green":
                                    if last_truck_alert:
                                        speak_queued("可通行，注意大型車", volume=1)
                                    else:
                                        speak_queued("可通行", volume=1)
                                last_traffic_color = color
                                last_speak_time = current_time
                            elif current_time - last_speak_time > speak_cooldown:
                                print("🔁 顏色相同但超過冷卻 → 再次播報")
                                if color == "Red":
                                    speak_queued("不可通行", volume=1)
                                elif color == "Yellow":
                                    speak_queued("不可通行", volume=1)
                                elif color == "Green":
                                    if last_truck_alert:
                                        speak_queued("可通行，注意大型車", volume=1)
                                    else:
                                        speak_queued("可通行", volume=1)
                                last_speak_time = current_time

                        
                        # 倒數秒數播報
                        try:
                            if countdown_number.isdigit():
                                countdown_value = int(countdown_number)
                                if countdown_value < 20 and not last_countdown_alert:
                                    print(f"⚠️ 播報提醒：秒數 {countdown_value} 低於 20 秒")
                                    speak_queued("請注意現在秒數低於20秒")
                                    detections.append("⏱️請注意現在秒數低於20秒")
                                    last_countdown_alert = True
                                elif countdown_value >= 30 and last_countdown_alert:
                                    print(f"🔄 重置 last_countdown_alert（當前秒數 {countdown_value}）")
                                    last_countdown_alert = False
                        except ValueError:
                            print("❌ 倒數秒數轉換失敗")

                        # 畫倒數秒數與框
                        cv2.putText(frame, f"{countdown_number}", (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # 顯示用文字
                        if color == "Red":
                            detections.append("🔴 紅燈 - 不可通行")
                        elif color == "Yellow":
                            detections.append("🟡 黃燈 - 不可通行")
                        elif color == "Green":
                            detections.append("🟢 綠燈 - 可通行")
                        else:
                            detections.append("⚪ 未知燈號")

        if not detections:
            detections.append("⚠️ 未偵測到紅綠燈及大型車，請注意周遭車輛")

        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frame = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("response_frame", {
            "image": encoded_frame,
            "detections": detections
        })

    except Exception as e:
        print(f"❌ 錯誤：{e}")


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)

