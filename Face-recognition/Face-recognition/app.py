import cv2
import numpy as np
import face_recognition
import mysql.connector
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time

# biến global
captured_image = None
frame = None
mode = 'recognition'  # Trạng thái ban đầu hiện tại của ứng dụng

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="spring",
        password="spring",
        database="testdb",
        port="3307"
    )

def create_table():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id INT AUTO_INCREMENT PRIMARY KEY,
        registration_time DATETIME NOT NULL,
        face_image LONGBLOB NOT NULL
    )
    """)
    db.commit()
    cursor.close()
    db.close()

def create_access_table():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS access_log (
        id INT AUTO_INCREMENT,
        student_id INT,
        access_time DATETIME NOT NULL,
        waste_type VARCHAR(255) NULL,
        waste_weight FLOAT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    )
    """)
    db.commit()
    cursor.close()
    db.close()

def insert_new_face(face_image):
    db = connect_db()
    cursor = db.cursor()
    registration_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    face_image_blob = cv2.imencode('.jpg', face_image)[1].tobytes()
    sql = "INSERT INTO students (registration_time, face_image) VALUES (%s, %s)"
    cursor.execute(sql, (registration_time, face_image_blob))
    db.commit()
    cursor.close()
    db.close()
    messagebox.showinfo("Success", f"Đã lưu vào cơ sở dữ liệu.")


# tải dự liệu trên csdl liệu về
def load_faces_from_db():
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT student_id, face_image FROM students")
    rows = cursor.fetchall()

    images = []
    student_ids = []

    for row in rows:
        student_id = row[0]
        face_image_blob = row[1]
        nparr = np.frombuffer(face_image_blob, np.uint8)
        face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(face_image)
        student_ids.append(student_id)

    cursor.close()
    db.close()

    return images, student_ids

# tiến hành encode face trong face_li
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encode_list.append(encode[0])
    return encode_list


# nhận dạng và đăng ký mặt mới (có 2 mode là recognize và register)
def recognize_and_register():
    global cap, mode, captured_image, frame
    curr_id = 0
    unknown_start_time = None

    def show_frame():
        nonlocal unknown_start_time, curr_id
        global cap, mode, captured_image, frame

        success, img = cap.read()
        if not success:
            return

        if mode == 'recognition':

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # tìm tạo độ khuôn mặt
            face_locations = face_recognition.face_locations(imgS)
            face_encodings = face_recognition.face_encodings(imgS, face_locations)


            for face_encoding, face_location in zip(face_encodings, face_locations):
                face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
                # flag để ghi nhận 1 gương mặt
                if len(face_distances) > 0:
                    matchIndex = np.argmin(face_distances)
                    if face_distances[matchIndex] < 0.6:
                        current_student_id = student_ids[matchIndex]
                        display_recognition_result(current_student_id, face_location, img)
                        if current_student_id != curr_id:
                            log_access(current_student_id, waste_type=None, waste_weight=None)
                            curr_id = current_student_id  # Sau khi log thì cập nhật User_detect để ngăn việc log lại

                    else:
                        display_unknown(face_location, img)
                        if unknown_start_time is None:
                            unknown_start_time = time.time()
                        elif time.time() - unknown_start_time > 5:
                            unknown_start_time = None
                            ask_for_registration()
                else:
                    display_unknown(face_location, img)
                    if unknown_start_time is None:
                        unknown_start_time = time.time()
                    elif time.time() - unknown_start_time > 5:
                        unknown_start_time = None
                        ask_for_registration()
        elif mode == 'registration':
            # Trong chế độ đăng ký, hiển thị khung hình bình thường
            pass

        # Hiển thị video trên Tkinter
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(imgRGB))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Lưu frame hiện tại để sử dụng khi chụp ảnh
        frame = img.copy()

        # Gọi lại chính hàm này để tiếp tục luồng video
        video_label.after(10, show_frame)

    show_frame()

def display_recognition_result(student_id, face_location, img):
    y1, x2, y2, x1 = [v * 4 for v in face_location]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'ID: {student_id}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

def display_unknown(face_location, img):
    y1, x2, y2, x1 = [v * 4 for v in face_location]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

# hiện cửa sổ hỏi user có đăng ký mặt mới không
def ask_for_registration():
    response = messagebox.askyesno("Unknown Face", "Khuôn mặt này chưa được nhận diện. Bạn có muốn đăng ký không?")
    if response:
        start_registration()


# lưu lại lịch sử tham gia ứng dụng
def log_access(student_id, waste_type=None, waste_weight=None):
    db = connect_db()
    cursor = db.cursor()
    access_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sql = "INSERT INTO access_log (student_id, access_time, waste_type, waste_weight) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (student_id, access_time, waste_type, waste_weight))
    db.commit()
    cursor.close()
    db.close()

def start_registration():
    global mode
    mode = 'registration'
    # Hiển thị các widget đăng ký
    capture_button.pack(pady=5)
    # Ẩn các widget nhận diện nếu cần

def back_to_recognition():
    global mode
    mode = 'recognition'
    # Ẩn các widget đăng ký
    capture_button.pack_forget()
    # Nếu muốn, có thể hiển thị lại các widget nhận diện

def capture_image():
    global captured_image, frame
    if frame is not None:
        captured_image = frame
        save_face()
    else:
        messagebox.showwarning("Warning", "Không có hình ảnh để lưu.")

def save_face():
    global captured_image, images, encodeListKnown, student_ids
    if captured_image is not None:
        captured_image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        encode_new_face = face_recognition.face_encodings(captured_image_rgb)

        if encode_new_face:
            insert_new_face(captured_image)
            images.append(captured_image)
            student_ids.append(len(student_ids) + 1)
            encodeListKnown.append(encode_new_face[0])
            messagebox.showinfo("Success", "Đăng ký khuôn mặt thành công!")
            back_to_recognition()
        else:
            messagebox.showwarning("Warning", "Không thể mã hóa khuôn mặt. Vui lòng thử lại.")
    else:
        messagebox.showwarning("Warning", "Không có hình ảnh để lưu.")

# Main window setup
create_table()
create_access_table()

root = tk.Tk()
root.title("Face Recognition System")
root_width = 800
root_height = 600
root.geometry(f"{root_width}x{root_height}")

# Video label
video_label = tk.Label(root, bg='white')
video_label.pack(fill=tk.BOTH, expand=True)

# Widgets for registration
capture_button = tk.Button(root, text="Chụp Ảnh", command=capture_image)

# Load known faces and encodings
images, student_ids = load_faces_from_db()
encodeListKnown = find_encodings(images)

# Initialize camera
cap = cv2.VideoCapture(0)

# Start the face recognition and registration process
recognize_and_register()

# Handle closing the application
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
