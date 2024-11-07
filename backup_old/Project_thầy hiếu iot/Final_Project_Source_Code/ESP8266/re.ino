#define BLYNK_TEMPLATE_ID "TMPL6Rx0f6hCW"
#define BLYNK_TEMPLATE_NAME "Based Waste Managerment System"
#define BLYNK_AUTH_TOKEN "QnKebZackpC8FsAXiUaiOPjPBtot3CK9"

#define BLYNK_PRINT Serial
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>

char auth[] = BLYNK_AUTH_TOKEN;
char ssid[] = "kiet1";
char pass[] = "123456789";

WidgetLCD lcd(V1); // Sử dụng LCD widget cho hiển thị mức độ đầy

void setup() {
  Serial.begin(9600);
  Blynk.begin(auth, ssid, pass);
}

void loop() {
  Blynk.run();

  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    if (data.startsWith("LEVEL:")) {
      int fillLevel = data.substring(6).toInt();
      Blynk.virtualWrite(V1, fillLevel); // Gửi mức độ đầy đến Blynk
    }
    if (data.startsWith("IR:OPEN")) {
      Blynk.virtualWrite(V2, 1); // Cập nhật trạng thái nút V2 là mở
    }
    if (data.startsWith("IR:CLOSE")) {
      Blynk.virtualWrite(V2, 0); // Cập nhật trạng thái nút V2 là đóng
    }
  }
}
