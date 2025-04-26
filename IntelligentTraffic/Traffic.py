import cv2
import numpy as np
import os
import easyocr
from ultralytics import YOLO
import collections
import pyttsx3 #引入語音播報
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import time
import pytesseract
import threading

reader = easyocr.Reader(['en'])  # 只使用英文來提高識別效率
engine = pyttsx3.init() # 初始化 pyttsx3 引擎
engine.setProperty('volume', 1) # 設定音量

# 避免語音重疊的全域變數
speaking = False
speaking_lock = threading.Lock() # 確保 speaking 變數的執行緒安全

def synchronized(func):
    """執行緒同步裝飾器"""
    def wrapper(*args, **kwargs):
        with speaking_lock:
            return func(*args, **kwargs)
    return wrapper

@synchronized
def text_to_speech(text, volume=1):
    """將文字轉換為語音並播放（避免重疊）"""
    global speaking
    if speaking:
        return # 如果正在說話，則直接返回
    
    speaking = True # 標記為正在說話
    engine.say(text) # 說話
    engine.runAndWait() # 等待說完
    speaking = False # 標記為已說完
    
def load_model(model_path='yolov8s.pt'):
    """載入YOLO模型"""
    try:
        model = YOLO(model_path)
        print("模型載入成功")
        return model
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return None
    
def is_daytime(frame):
    """判斷當前場景是白天還是晚上"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > 80  # 亮度閾值（需根據實測調整）

def detect_traffic_light_color(roi, is_daytime_scene):
    """偵測交通燈的顏色"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定義白天與晚上的 HSV 範圍
    if is_daytime_scene:
        lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
        lower_yellow, upper_yellow = np.array([15, 120, 120]), np.array([35, 255, 255])
        lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])
    else:
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        lower_yellow, upper_yellow = np.array([15, 50, 50]), np.array([35, 255, 255])
        lower_green, upper_green = np.array([30, 50, 50]), np.array([120, 255, 255])

    # 定義方向燈的 HSV 範圍
    lower_arrow_green, upper_arrow_green = np.array([35, 100, 100]), np.array([85, 255, 255])
    lower_arrow_red, upper_arrow_red = np.array([0, 100, 100]), np.array([10, 255, 255])

    # 計算顏色遮罩
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    arrow_green_mask = cv2.inRange(hsv, lower_arrow_green, upper_arrow_green)
    arrow_red_mask = cv2.inRange(hsv, lower_arrow_red, upper_arrow_red)

    # 計算區域面積
    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    green_area = cv2.countNonZero(green_mask)
    arrow_green_area = cv2.countNonZero(arrow_green_mask)
    arrow_red_area = cv2.countNonZero(arrow_red_mask)

    roi_size = roi.shape[0] * roi.shape[1]
    if red_area / roi_size < 0.05: red_area = 0
    if yellow_area / roi_size < 0.05: yellow_area = 0
    if green_area / roi_size < 0.05: green_area = 0

    # 判斷顏色
    if arrow_red_area > 0:
        return "Red"
    elif arrow_green_area > 0:
        return "Green"
    elif red_area > yellow_area and red_area > green_area:
        return "Red"
    elif yellow_area > red_area and yellow_area > green_area:
        return "Yellow"
    elif green_area > red_area and green_area > yellow_area:
        return "Green"
    return "Unknown"

def determine_side(roi, is_daytime_scene):
    """判斷紅綠燈上的箭頭方向（左、右或直行）"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定義白天與晚上的 HSV 範圍
    if is_daytime_scene:
        lower_arrow_green, upper_arrow_green = np.array([35, 100, 100]), np.array([85, 255, 255])
        lower_arrow_red, upper_arrow_red = np.array([0, 100, 100]), np.array([10, 255, 255])
    else:
        lower_arrow_green, upper_arrow_green = np.array([35, 50, 50]), np.array([85, 255, 255])
        lower_arrow_red, upper_arrow_red = np.array([0, 50, 50]), np.array([10, 255, 255])

    # 計算箭頭遮罩
    green_mask = cv2.inRange(hsv, lower_arrow_green, upper_arrow_green)
    red_mask = cv2.inRange(hsv, lower_arrow_red, upper_arrow_red)

    # 使用形態學操作加強箭頭形狀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # 檢測箭頭方向
    contours, _ = cv2.findContours(green_mask | red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)  # 計算寬高比以區分箭頭形狀
        if 0.3 < aspect_ratio < 0.6:  # 判斷寬高比是否可能是箭頭形狀
            center_x = x + w // 2
            if center_x < roi.shape[1] // 2:
                return "Left"
            else:
                return "Right"

    return "Straight"

def extract_traffic_light_number(roi, is_daytime_scene):
    """根據時間條件選擇 OCR 方法來辨識交通燈號秒數"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定義黃色範圍
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    # 過濾出黃色區域
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_part = cv2.bitwise_and(roi, roi, mask=mask)

    # 轉灰階
    gray = cv2.cvtColor(yellow_part, cv2.COLOR_BGR2GRAY)

    # 直方圖均衡化，使影像亮度更均勻
    gray = cv2.equalizeHist(gray)

    # 高斯模糊去除雜訊
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 嘗試二值化 (Otsu’s Thresholding)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if is_daytime_scene:
        # 使用 EasyOCR
        result = reader.readtext(binary, allowlist="0123456789", detail=0)
    else:
        # 使用 Tesseract OCR
        result = pytesseract.image_to_string(binary, config="--psm 6 digits") # 過濾出數字
    text = ''.join(filter(str.isdigit, ''.join(result)))

    return text if text else "N/A"

def process_frame(model, frame):
    """即時處理攝影機影像，並回傳 YOLO 偵測結果"""
    results = model(frame)
    detections = []
    
    # 用於避免重複語音播報
    last_traffic_color = "Unknown"
    last_truck_alert = False
    last_countdown_alert = False
    result_queue = collections.deque(maxlen=5)

    is_daytime_scene = is_daytime(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            class_id = int(box.cls)
                
            if conf > 0.3:
                if class_id == 7:  # Assuming class 7 is for trucks, this can change based on your model's class IDs
                    box_color = (0, 0, 255)  # 紅色框
                    label = f"Truck {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)  # 繪製紅色框
                        
                    if not last_truck_alert:  # Ensure the alert is only triggered once for each truck detection
                        text_to_speech("小心大型車", volume=1)
                        last_truck_alert = True  # Set the flag to True after alerting
                    
            if conf > 0.3 and class_id in [9, 10, 11]:  # 9: 普通燈, 10: 左箭頭, 11: 右箭頭
                traffic_light_roi = frame[y1:y2, x1:x2]
                if traffic_light_roi.size > 0:
                    color = detect_traffic_light_color(traffic_light_roi, is_daytime_scene)
                    side = determine_side(traffic_light_roi, is_daytime_scene)  # 判斷左右邊

                    result_queue.append((color, side))
                    recent_results = [r[0] for r in result_queue if r[1] == side]
                    if len(recent_results) > 2:  
                        if recent_results.count("Red") > 2:
                            color = "Red"
                        elif recent_results.count("Yellow") > 2:
                            color = "Yellow"
                        elif recent_results.count("Green") > 3:
                            color = "Green"
                        else:
                            color = "Unknown"
                    else:
                        color = "Unknown"  # 如果沒有足夠數據，則不判定
                        
                    box_color = (0, 255, 0) if "Green" in color else \
                                    (0, 255, 255) if "Yellow" in color else \
                                    (0, 0, 255) if "Red" in color else \
                                    (255, 255, 255)
                    label = f"{color} {side} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                    countdown_number = extract_traffic_light_number(traffic_light_roi, is_daytime_scene)

                    if color == "Unknown" and color == last_traffic_color:
                        text_to_speech("未偵測到，請注意周遭車輛", volume=0.2)
                    elif color  != "Unknown" and color != last_traffic_color:
                        if color == "Red":
                            text_to_speech("不可通行", volume=1)
                            last_traffic_color = color
                        elif color == "Yellow":
                            text_to_speech("不可通行", volume=1)
                            last_traffic_color = color
                        elif last_truck_alert: #先判斷卡車再判斷綠燈
                            text_to_speech("可通行，注意大型車", volume=1)
                            last_traffic_color = color
                        elif color == "Green":
                            text_to_speech("可通行", volume=1)
                            last_traffic_color = color
                        elif color == "Unknown":
                            text_to_speech("未偵測到，請注意周遭車輛", volume=0.2)
                            last_traffic_color = color
                        
                    # 添加一個計數器來記錄低於 20 秒的次數
                countdown_below_20_count = 0
                    # 倒數時間低於 20 秒時播報一次
                try:
                    if countdown_number.isdigit():  # 確保 countdown_number 是有效數字
                        countdown_value = int(countdown_number)
                    else:
                        countdown_value = None  # 若不是數字，設定為 None

                    if countdown_value is not None and countdown_value < 20 and not last_countdown_alert:
                        print(f"⚠️ 播報提醒：秒數 {countdown_value} 低於 20 秒")  
                        text_to_speech("請注意現在秒數低於20秒")
                        last_countdown_alert = True  

                    elif countdown_value is not None and countdown_value >= 30 and last_countdown_alert:
                        print(f"🔄 重置 last_countdown_alert（當前秒數 {countdown_value}）")  
                        last_countdown_alert = False
                except ValueError:
                    countdown_value = None  # 解析錯誤則忽略

                    # Display the countdown number
                cv2.putText(frame, f"{countdown_number}", (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                    # 繪製框
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    return frame, detections

if __name__ == "__main__":
    model_path =  "yolov8s.pt"
    model = YOLO(model_path)
    
    print("✅ 模型載入成功，請使用 `process_frame(model, frame)` 來即時處理影像。")


