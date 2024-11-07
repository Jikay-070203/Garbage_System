import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date, timedelta
import threading

# Đường dẫn đến thư mục chứa ảnh
path = 'Images_Excel'
# Danh sách lưu trữ ảnh và tên
images = []
classNames = []

# Lấy danh sách tên file trong thư mục
myList = os.listdir(path)
print(myList)

# Đọc ảnh từ thư mục và lưu vào danh sách images, tên tương ứng vào classNames
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Hàm mã hóa ảnh thành vector đặc trưng
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if encoding is successful
            encodeList.append(encode[0])
            print("encode là", encode[0])
    return encodeList

# Hàm tạo tên file CSV theo ngày
def get_csv_filename():
    today = date.today()
    return f"Danh_Sach_Tham_Gia_{today.strftime('%d-%m-%Y')}.csv"

# Hàm tạo file CSV, nếu file đã tồn tại thì thông báo
def create_csv_file():
    filename = get_csv_filename()
    if os.path.isfile(filename):
        print(f"CSV file already exists: {filename}")
    else:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Name,Time,Valmin')
        print(f"CSV file created: {filename}")
    return filename

# Hàm ghi thông tin điểm danh vào file CSV
def Attendance(name, valmin, csv_file):
    with open(csv_file, 'a', encoding='utf-8') as f:
        now = datetime.now()
        dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
        f.write(f'\n{name},{dtString},{valmin}')

# Mã hóa toàn bộ ảnh trong thư mục
encodeListKnown = findEncodings(images)
print('Encoding OK. Loading camera...')

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Tạo file CSV
csv_file = create_csv_file()
print(f"CSV file created: {csv_file}")

# Danh sách lưu trữ khuôn mặt đã điểm danh và thời gian điểm danh
attending_faces = {}
displayed_faces = {}
reappearance_interval = timedelta(seconds=10)  # Thời gian để nhận diện lại
display_duration = timedelta(seconds=10)  # Thời gian để hiển thị khung và tên

def recognize_faces():
    global attending_faces, displayed_faces
    while True:
        # Đọc ảnh từ camera
        success, img = cap.read()
        if not success:
            break
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

        # Xóa những khuôn mặt không còn cần thiết để hiển thị
        to_remove = [name for name, timestamp in displayed_faces.items() 
                     if name not in current_faces_displayed or 
                     datetime.now() - timestamp > display_duration]
        for name in to_remove:
            del displayed_faces[name]

        # Hiển thị ảnh
        cv2.imshow('Camera 01', img)
        # Thoát chương trình khi bấm phím 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()


# Bắt đầu nhận diện khuôn mặt trong một luồng riêng
recognition_thread = threading.Thread(target=recognize_faces)
recognition_thread.start()

# Hàm lưu vector đặc trưng vào file
def save_encodings_to_file(encodings, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for encoding in encodings:
            np.savetxt(f, encoding, newline=' ')
            f.write('\n')

# Lưu vector đặc trưng vào file encodings.txt
save_encodings_to_file(encodeListKnown, 'encodings.txt')
