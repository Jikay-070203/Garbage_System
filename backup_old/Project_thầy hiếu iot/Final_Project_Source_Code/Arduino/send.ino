#include <Servo.h>
#include <NewPing.h>

#define TRIGGER_PIN  9
#define ECHO_PIN     8
#define MAX_DISTANCE 200

#define SERVO_PIN 7
#define IR_SENSOR_PIN 6

Servo myservo;
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

unsigned long lastIRCheckTime = 0;
const unsigned long IRCheckInterval = 500; // Kiểm tra cảm biến mỗi 500ms
const int binHeight = 15; // Chiều cao của thùng rác (cm)

void setup() {
  Serial.begin(9600); // Sử dụng Serial cứng để giao tiếp với ESP8266

  myservo.attach(SERVO_PIN);
  myservo.write(0); // Cửa thùng rác đóng
  
  pinMode(IR_SENSOR_PIN, INPUT);
}

void loop() {
  // Đọc khoảng cách từ cảm biến siêu âm
  int distance = sonar.ping_cm();
  int fillLevel = 0;

  if (distance >= 0 && distance <= binHeight) {
    fillLevel = map(distance, binHeight, 0, 0, 100); // Tính phần trăm mức độ đầy
  } else if (distance > binHeight) {
    fillLevel = 0; // Nếu khoảng cách lớn hơn chiều cao thùng rác, nghĩa là thùng rác trống
  }

  // Giới hạn giá trị phần trăm từ 0 đến 100
  fillLevel = constrain(fillLevel, 0, 100);

  // Gửi mức độ đầy đến ESP8266
  Serial.print("LEVEL:");
  Serial.println(fillLevel);

  // Kiểm tra cảm biến hồng ngoại (IR sensor)
  unsigned long currentMillis = millis();
  if (currentMillis - lastIRCheckTime >= IRCheckInterval) {
    lastIRCheckTime = currentMillis;
    if (digitalRead(IR_SENSOR_PIN) == LOW) {
      myservo.write(0); // Mở cửa thùng rác khi phát hiện người đến gần
      Serial.println("IR:OPEN"); // Gửi tín hiệu mở cửa đến ESP8266
    } else {
      myservo.write(160); // Đóng cửa thùng rác khi người đi khỏi
      Serial.println("IR:CLOSE"); // Gửi tín hiệu đóng cửa đến ESP8266
    }
  }

  delay(100); // Kiểm tra mỗi 0.1 giây
}
