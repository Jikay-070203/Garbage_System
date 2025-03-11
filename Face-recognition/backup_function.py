import cv2
import numpy as np
import face_recognition
import mysql.connector
from datetime import datetime


# Kết nối tới MySQL
def connect_db():
    """Kết nối đến cơ sở dữ liệu MySQL"""
    return mysql.connector.connect(
        host="localhost",
        user="spring",
        password="spring",
        database="testdb",
        port="3307"
    )


def create_table():
    """Tạo bảng students nếu chưa tồn tại."""
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        registration_time DATETIME NOT NULL,
        face_image LONGBLOB NOT NULL
    )
    """)
    db.commit()
    cursor.close()
    db.close()


def insert_new_face(name, face_image):
    """Lưu ảnh khuôn mặt và tên vào cơ sở dữ liệu."""
    db = connect_db()
    cursor = db.cursor()
    registration_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Chuyển đổi ảnh khuôn mặt thành BLOB
    face_image_blob = cv2.imencode('.jpg', face_image)[1].tobytes()

    sql = "INSERT INTO students (name, registration_time, face_image) VALUES (%s, %s, %s)"
    cursor.execute(sql, (name, registration_time, face_image_blob))
    db.commit()
    cursor.close()
    db.close()
    print(f"Đã lưu {name} vào cơ sở dữ liệu.")


def load_faces_from_db():
    """Tải ảnh và tên từ cơ sở dữ liệu, trả về list ảnh và tên."""
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, name, face_image FROM students")
    rows = cursor.fetchall()

    images = []
    student_ids = []
    classNames = []

    for row in rows:
        student_id = row[0]
        name = row[1]
        face_image_blob = row[2]

        # Chuyển đổi BLOB sang hình ảnh
        nparr = np.frombuffer(face_image_blob, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        images.append(face_image)
        classNames.append(name)
        student_ids.append(student_id)

    cursor.close()
    db.close()

    return images, classNames, student_ids


def find_encodings(images):
    """Mã hóa danh sách ảnh thành vector đặc trưng."""
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encode_list.append(encode[0])
    return encode_list


def recognize_face(encodeListKnown, student_ids):
    """Nhận diện khuôn mặt hiện tại và trả về ID sinh viên nếu có."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(imgS)
        face_encodings = face_recognition.face_encodings(imgS, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            if len(face_distances) > 0:
                matchIndex = np.argmin(face_distances)

                # Kiểm tra nếu khuôn mặt trùng khớp với ID trong database
                if face_distances[matchIndex] < 0.6:
                    student_id = student_ids[matchIndex]
                    display_recognition_result(student_id, face_location, img)
                else:
                    display_unknown(face_location, img)

        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def display_recognition_result(student_id, face_location, img):
    """Hiển thị kết quả nhận diện với ID của sinh viên."""
    y1, x2, y2, x1 = [v * 4 for v in face_location]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'ID: {student_id}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def display_unknown(face_location, img):
    """Hiển thị khung và thông báo Unknown khi không nhận diện được."""
    y1, x2, y2, x1 = [v * 4 for v in face_location]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)


def register_new_face():
    """Chụp ảnh và đăng ký khuôn mặt mới."""
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break

        cv2.imshow('Camera - Press S to Save', img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            name = input("Enter name for the new face: ")
            insert_new_face(name, img)
            break

    cap.release()
    cv2.destroyAllWindows()


# Khởi động chương trình
if __name__ == "__main__":
    create_table()

    # Tải dữ liệu từ cơ sở dữ liệu
    images, classNames, student_ids = load_faces_from_db()
    encodeListKnown = find_encodings(images)

    print("Chọn option:")
    print("1. Nhận diện khuôn mặt")
    print("2. Đăng ký khuôn mặt mới")

    option = input("Nhập lựa chọn: ")

    if option == '1':
        recognize_face(encodeListKnown, student_ids)
    elif option == '2':
        register_new_face()
        # Sau khi đăng ký khuôn mặt, quay lại nhận diện
        images, classNames, student_ids = load_faces_from_db()
        encodeListKnown = find_encodings(images)
        recognize_face(encodeListKnown, student_ids)
