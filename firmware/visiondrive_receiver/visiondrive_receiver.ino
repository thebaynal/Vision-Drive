// Laptop vision -> Wi-Fi commands -> ESP32 motor controller.
// Supports Arduino-ESP32 core 2/3. No camera or ML model runs on this board.
#include <WiFi.h>
#include <WiFiUdp.h>
#include "command_state.h"
#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

WiFiUDP udp;
visiondrive::CommandState commands(COMMAND_TOKEN);
visiondrive::Action applied = visiondrive::Action::Stop;
bool motorPinsReady = false;

void wheel(int pwm, int in1, int in2, bool forward) {
    analogWrite(pwm, 0);
    digitalWrite(in1, forward ? HIGH : LOW);
    digitalWrite(in2, LOW);
    if (forward) analogWrite(pwm, MOTOR_DUTY);
}

void apply(visiondrive::Action action) {
    if (action == applied) return;
    if (ENABLE_MOTORS && motorPinsReady) {
        bool left = action == visiondrive::Action::Forward || action == visiondrive::Action::Right;
        bool right = action == visiondrive::Action::Forward || action == visiondrive::Action::Left;
        wheel(LEFT_PWM, LEFT_IN1, LEFT_IN2, left);
        wheel(RIGHT_PWM, RIGHT_IN1, RIGHT_IN2, right);
    }
    if (action != applied) Serial.println(visiondrive::name(action));
    applied = action;
}

void setup() {
    Serial.begin(115200);
    if (ENABLE_MOTORS) {
        const int pins[] = {LEFT_PWM, LEFT_IN1, LEFT_IN2, RIGHT_PWM, RIGHT_IN1, RIGHT_IN2};
        bool valid = MOTOR_DUTY > 0 && MOTOR_DUTY <= 100;
        for (int i = 0; i < 6; ++i) {
            valid &= pins[i] >= 0;
            for (int j = 0; j < i; ++j) valid &= pins[i] != pins[j];
        }
        if (!valid) { Serial.println("STOP: configure valid unique motor pins/duty"); return; }
        for (int pin : pins) { pinMode(pin, OUTPUT); digitalWrite(pin, LOW); }
        motorPinsReady = true;
    }
    apply(visiondrive::Action::Stop);
    if (strlen(COMMAND_TOKEN) != 32 || strspn(COMMAND_TOKEN, "0123456789abcdefABCDEF") != 32) {
        Serial.println("STOP: copy config.example.h to config.h and set credentials/token");
        return;
    }
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.println("Waiting for Wi-Fi; motors stopped");
}

void loop() {
    static bool listening = false;
    commands.tick(millis());
    if (WiFi.status() != WL_CONNECTED) {
        commands.disconnect();
        apply(visiondrive::Action::Stop);
        if (listening) { udp.stop(); listening = false; }
        delay(2);
        return;
    }
    if (!listening) {
        listening = udp.begin(COMMAND_PORT);
        Serial.print("ESP32 address: "); Serial.println(WiFi.localIP());
        if (!listening) { apply(visiondrive::Action::Stop); delay(20); return; }
    }
    // Limit packet work per loop so traffic cannot starve the watchdog.
    for (int i = 0; i < 4; ++i) {
        int size = udp.parsePacket();
        if (size <= 0) break;
        char packet[128];
        int length = udp.read(packet, sizeof(packet)-1);
        if (size >= int(sizeof(packet)) || length <= 0) continue;
        packet[length] = '\0';
        if (strlen(packet) != size_t(length)) continue;
        uint64_t sender = (uint64_t(uint32_t(udp.remoteIP())) << 16) | udp.remotePort();
        if (commands.accept(packet, sender, millis())) {
            char reply[96];
            snprintf(reply, sizeof(reply), "VA1 %s %lu %s", commands.session,
                     (unsigned long)commands.sequence, visiondrive::name(commands.action));
            udp.beginPacket(udp.remoteIP(), udp.remotePort());
            udp.write((uint8_t *)reply, strlen(reply));
            udp.endPacket();
        }
        commands.tick(millis());
        apply(commands.action);
    }
    commands.tick(millis());
    apply(commands.action);
    delay(2);
}
