#define BAUD_RATE 460800

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) {
    ;
  }

  // confi ADC
  analogReadResolution(14);

  // config pins
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
}

void loop() {
  int ch1Val = analogRead(A0);
  int ch2Val = analogRead(A1);
  int ch3Val = analogRead(A2);

  // data transform
  byte ch1H = (ch1Val >> 8) & 0xFF;
  byte ch1L = ch1Val & 0xFF;

  byte ch2H = (ch2Val >> 8) & 0xFF;
  byte ch2L = ch2Val & 0xFF;

  byte ch3H = (ch3Val >> 8) & 0xFF;
  byte ch3L = ch3Val & 0xFF;

  // sending
  byte dataToSend[8] = { 0x7F, 0xFF, ch1H, ch1L, ch2H, ch2L, ch3H, ch3L };
  Serial.write(dataToSend, 8);
}
