#pragma once
// Copy to config.h, which is ignored by Git. Both devices use the same Wi-Fi.
constexpr const char *WIFI_SSID = "YOUR_WIFI_NAME";
constexpr const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Generate: python -c "import secrets; print(secrets.token_hex(16))"
constexpr const char *COMMAND_TOKEN = "REPLACE_WITH_32_HEXADECIMAL_CHARS";
constexpr unsigned short COMMAND_PORT = 4210;

// First verify instructions in Serial Monitor with motors disabled.
// Use two PWM+direction channels on a compatible H-bridge, e.g. TB6612FNG.
// Select valid, free GPIOs for YOUR exact board before enabling motor output.
constexpr bool ENABLE_MOTORS = false;
constexpr int LEFT_PWM = -1, LEFT_IN1 = -1, LEFT_IN2 = -1;
constexpr int RIGHT_PWM = -1, RIGHT_IN1 = -1, RIGHT_IN2 = -1;
// Wire the driver's STBY/ENABLE separately as required by its datasheet.
constexpr int MOTOR_DUTY = 65; // Out of 255; calibrate speed and braking.
