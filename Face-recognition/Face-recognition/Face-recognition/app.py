from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
from datetime import datetime, date, timedelta

app = Flask(__name__)

# Đường dẫn tới thư mục ảnh và CSV
path = 'Images_Excel'
csv_file = None
encodeListKnown = []
    

# Hàm mã hóa ảnh
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

# Load và mã hóa ảnh ban đầu
def load_images_and_encodings():
    global encodeListKnown, classNames
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)
    return classNames

classNames = load_images_and_encodings()

# Tạo file CSV mới
def create_csv_file():
    global csv_file
    filename = datetime.now().strftime("Danh_Sach_Tham_Gia_%d-%m-%Y.csv")
    if not os.path.isfile(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Name,Time,Valmin')
    csv_file = filename


@app.route('/')
def index():
    return render_template('index.html')

# Endpoint: Nhận diện khuôn mặt từ hình ảnh tải lên
@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    img = face_recognition.load_image_file(file)
    encodesCurFrame = face_recognition.face_encodings(img)
    
    if len(encodesCurFrame) == 0:
        return jsonify({"message": "No face found"})
    
    encodeFace = encodesCurFrame[0]
    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    matchIndex = np.argmin(faceDis)
    if faceDis[matchIndex] < 0.6:
        name = classNames[matchIndex].upper()
        valmin = round(100 * (1 - faceDis[matchIndex]))
        Attendance(name, valmin, csv_file)
        return jsonify({"name": name, "accuracy": f"{valmin}%"})
    else:
        return jsonify({"message": "No match found"})

# Endpoint: Lấy danh sách điểm danh
@app.route('/attendance', methods=['GET'])
def get_attendance():
    if csv_file and os.path.isfile(csv_file):
        return send_file(csv_file)
    return jsonify({"message": "No attendance record found"})

# Endpoint: Lưu vector đặc trưng vào file
@app.route('/save_encodings', methods=['POST'])
def save_encodings():
    data = request.json.get('encodings', [])
    if not data:
        return jsonify({"error": "No encodings provided"})
    
    with open('encodings.txt', 'w', encoding='utf-8') as f:
        for encoding in data:
            np.savetxt(f, np.array(encoding), newline=' ')
            f.write('\n')
    return jsonify({"message": "Encodings saved successfully"})

# Endpoint: Lấy danh sách các hình ảnh đã mã hóa và tên
@app.route('/encodings', methods=['GET'])
def get_encodings():
    return jsonify({"images": classNames, "encodings": encodeListKnown})

# Endpoint: Khởi động nhận diện khuôn mặt trong thời gian thực
@app.route('/start_recognition', methods=['GET'])
def start_recognition():
    threading.Thread(target=recognize_faces).start()
    return jsonify({"message": "Recognition started"})

# Hàm ghi thông tin điểm danh vào file CSV
def Attendance(name, valmin, csv_file):
    with open(csv_file, 'a', encoding='utf-8') as f:
        now = datetime.now()
        dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
        f.write(f'\n{name},{dtString},{valmin}')
        
        
def recognize_faces():
    global attending_faces, displayed_faces
    cap = cv2.VideoCapture(0)
    attending_faces = {}
    displayed_faces = {}
    reappearance_interval = timedelta(seconds=10)  # Thời gian để nhận diện lại
    display_duration = timedelta(seconds=10)  # Thời gian để hiển thị khung và tên
    while True:
        success, img = cap.read()
        if not success:
            break
        # ... [Mã nhận diện khuôn mặt thời gian thực từ script gốc]
        # Resize ảnh để tăng tốc độ xử lý
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # Chuyển đổi ảnh sang không gian màu RGB
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt trong ảnh
        facesCurFrame = face_recognition.face_locations(imgS)
        # Mã hóa khuôn mặt trong ảnh
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Lưu danh sách các khuôn mặt đang hiển thị
        current_faces_displayed = set()

        # Duyệt qua từng khuôn mặt được phát hiện
        for i, encodeFace in enumerate(encodesCurFrame):
            # Lấy vị trí khuôn mặt
            faceLoc = facesCurFrame[i]
            # So sánh khuôn mặt hiện tại với danh sách khuôn mặt đã biết
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            # Tính khoảng cách giữa khuôn mặt hiện tại và danh sách khuôn mặt đã biết
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            
            # Điều chỉnh ngưỡng (threshold) cho nhận diện chính xác hơn
            threshold = 0.6  # Bạn có thể điều chỉnh giá trị này
            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex] < threshold:
                matches[matchIndex] = True
            else:
                matches = [False] * len(matches)

            # Nếu khuôn mặt được nhận dạng
            if any(matches):
                # Lấy tên và độ chính xác
                name = classNames[matchIndex].upper()
                valmin = "{}".format(round(100 * (1 - faceDis[matchIndex])))

                # Nếu khuôn mặt chưa được điểm danh hoặc thời gian điểm danh đã hết
                if (name not in attending_faces or
                    datetime.now() - attending_faces[name] > reappearance_interval):
                    
                    # Ghi thông tin điểm danh vào file CSV
                    Attendance(name, valmin, csv_file)
                    
                    # Cập nhật thời gian điểm danh
                    attending_faces[name] = datetime.now()

                # Cập nhật thời gian hiển thị khuôn mặt
                displayed_faces[name] = datetime.now()
                current_faces_displayed.add(name)

                # Vẽ hình chữ nhật và tên lên ảnh
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name + ' - ' + valmin + '%', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        
        to_remove = [name for name, timestamp in displayed_faces.items() if name not in current_faces_displayed or datetime.now() - timestamp > display_duration]
        for name in to_remove:
            del displayed_faces[name]

        # Hiển thị ảnh
        cv2.imshow('Camera 01', img)
        # Thoát chương trình khi bấm phím 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    create_csv_file()
    app.run(debug=True)
    
    
