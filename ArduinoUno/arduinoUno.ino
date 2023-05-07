//-----------------------------------------------------------------------------------------------------------------------------------
// Librerias, variables y constantes
//-----------------------------------------------------------------------------------------------------------------------------------
#include <Servo.h>

#define TORNILLO 0
#define TUERCA 1
#define ARANDELA 2
#define MARIPOSA 3

// Pines del motor paso a paso 
#define IN1 10
#define IN2 11
#define IN3 12
#define IN4 13

// Servos
Servo servoRL;
Servo servoCaja;
Servo servoCinta;

// Contador de grados que se mueve el brazo
int contRL = 90;

// Contador de grados que se mueve la cesta del brazo
int contCaja = 0;

// Secuencia de pasos (par máximo)
int paso[8][4] = {
  { 1, 0, 0, 0 },
  { 1, 1, 0, 0 },
  { 0, 1, 0, 0 },
  { 0, 1, 1, 0 },
  { 0, 0, 1, 0 },
  { 0, 0, 1, 1 },
  { 0, 0, 0, 1 },
  { 1, 0, 0, 1 }
};

//Pin de boton de entrada de datos
int botonPin = 2;
int boton2Pin = 6;
int buttonState;
int buttonState2;

// Pines del joystick y angulos 
int x_pin = A0;
int y_pin = A1;
int anguloX = 0;
int anguloY = 0;

//resultado del metodo identificar
int resultado;

//-----------------------------------------------------------------------------------------------------------------------------------
// Iniciacion y bucle de ejecucion
//-----------------------------------------------------------------------------------------------------------------------------------
void setup() {
  //Asiganacion de entradas y salida de pines
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(botonPin, INPUT);

  // Asignamos el pin a los servos
  servoRL.attach(4);
  servoCaja.attach(3);
  servoCinta.attach(9);

  // Posicion inicial de los servos
  servoRL.write(contRL);
  servoCaja.write(contCaja);
  
  // Iniciamos el serial
  Serial.begin(115200);
}

void loop() {

  buttonState = digitalRead(botonPin);

  while (buttonState == HIGH) {
    servoCinta.write(180);
    buttonState = digitalRead(botonPin);
  }

  while (buttonState == LOW) {
    servoCinta.write(92);
    buttonState = digitalRead(botonPin);
    buttonState2 = digitalRead(boton2Pin);
   
    if (buttonState2 == HIGH) {

      Serial.println("Llamada a la ia para clasificar...");

      Serial.print("Loop 1\n"); 

      resultado = identificar();

      delay(500);

      Serial.print("El numero recibido es...");
      Serial.println(resultado);

      delay(500);
      Serial.println("Movimiento del brazo para soltar y volver atras...");

      while (contRL < 180) {
        contRL++;
        servoRL.write(contRL);
        delay(3);
      }

      servoCinta.write(180);
      delay(2350);
      servoCinta.write(92);
      delay(500);
      servoCinta.write(0);
      delay(800);
      servoCinta.write(92);
      organizar(resultado);
    }
  }
}// FIN DE PROGRAMA

//-----------------------------------------------------------------------------------------------------------------------------------
// Funciones Auxiliares
//-----------------------------------------------------------------------------------------------------------------------------------
void mover_brazo() {
  contRL = 180;
  while (contRL > 90) {
    contRL--;
    servoRL.write(contRL);
    delay(8);
  }
  delay(500);

  // Mover caja para descargar objeto
  while (contCaja < 165) {
    contCaja++;
    servoCaja.write(contCaja);
    delay(10);
  }
  delay(250);
  while (contCaja > 0) {
    contCaja--;
    servoCaja.write(contCaja);
    delay(10);
  }
  delay(250);
}

//-----------------------------------------------------------------------------------------------------------------------------------
void organizar(int n) {
  //Comprobar que es lo que hay que organizar
  switch (n) {

    case 1:
      organizar_tuerca();
      break;

    case 2:
      organizar_arandela();
      break;

    case 3:
      organizar_mariposa();
      break;

    default:
      organizar_tornillo();
      break;
  }
}

//-----------------------------------------------------------------------------------------------------------------------------------
void organizar_tornillo() {
  // NO mover rueda de cajas (estado por defecto)
  mover_brazo();
}

void organizar_tuerca() {
  mover_cajas();
  mover_brazo();
  for (int i = 0; i < 3; i++) {
    mover_cajas();
  }
}

//-----------------------------------------------------------------------------------------------------------------------------------
void organizar_arandela() {
  for (int i = 0; i < 2; i++) {
    mover_cajas();
  }
  mover_brazo();
  for (int i = 0; i < 2; i++) {
    mover_cajas();
  }
}

//-----------------------------------------------------------------------------------------------------------------------------------
void organizar_mariposa() {
  for (int i = 0; i < 3; i++) {
    mover_cajas();
  }
  mover_brazo();
  mover_cajas();
}

//-----------------------------------------------------------------------------------------------------------------------------------
void mover_cajas() {
  int j = 0;

  while (j < 130) {  //gira un cuarto de vuelta las cajas clasificadoras
    j++;
    for (int i = 0; i < 8; i++) {
      digitalWrite(IN1, paso[i][0]);
      digitalWrite(IN2, paso[i][1]);
      digitalWrite(IN3, paso[i][2]);
      digitalWrite(IN4, paso[i][3]);
      delay(2);
    }
  }
}

//-----------------------------------------------------------------------------------------------------------------------------------
int identificar() {
  int objClasificado = -1;

  while (objClasificado == -1) {

    int valorX = analogRead(x_pin);  
    int valorY = analogRead(y_pin);

    //Lo pasamos a valores en angulo (de 0 a 180 grados)
    anguloX = map(valorX, 0, 1023, 0, 180);  //args -> valorTrans, minRangIni, macRangIni, ...
    anguloY = map(valorY, 0, 1023, 0, 180);

    delay(300);  //Para que no nos imprima valores a cada rato
    if (anguloX == 87 && anguloY == 180) {
      Serial.println("Tornillo");
      objClasificado = TORNILLO;
    } else if (anguloX == 180 && anguloY == 88) {
      Serial.println("Tuerca");
      objClasificado = TUERCA;
    } else if (anguloX == 87 && anguloY == 0) {
      Serial.println("Arandela");
      objClasificado = ARANDELA;
    } else if (anguloX == 0 && anguloY == 88) {
      Serial.println("Mariposa");
      objClasificado = MARIPOSA;
    }
  }
  return objClasificado;
}
//-----------------------------------------------------------------------------------------------------------------------------------
// Final
//-----------------------------------------------------------------------------------------------------------------------------------