import cv2
import torch
import pygame
import os
import telegram
import asyncio
from datetime import datetime

# Configura tu token de bot
#TOKEN = TOKEN DEL BOT
bot = telegram.Bot(token=TOKEN)

# Función asincrónica para enviar una imagen
async def send_image(file_image):
    if os.path.exists(file_image):
        # Obtener la fecha y hora actual
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")

        with open(file_image, 'rb') as img:
            caption = f'Detección de fuego el {date_time}'
            await bot.send_photo(chat_id='-4175576779', photo=img, caption=caption)

async def main():
    # Inicializa pygame
    pygame.init()

    # Carga tu modelo preentrenado en formato .pt
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='helmet-yolov5s.pt')

    # Configura el modelo en modo de evaluación
    model.eval()

    # Inicializa la cámara web
    cap = cv2.VideoCapture(0)

    # Contador para analizar solo cada 3 fotogramas
    frame_count = 0
    frame_skip = 2  # Analizar cada 3er fotograma

    # Carga el sonido de alarma
    pygame.mixer.music.load('alarm.mp3')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (frame_skip + 1) != 0:
            continue

        # Detecta objetos en el fotograma
        results = model(frame)

        # Dibuja los cuadros delimitadores en el video
        threshold = 0.40
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Reproduce el sonido de alarma
                pygame.mixer.music.play()
                # Guarda la imagen
                cv2.imwrite('imagen_detectada.jpg', frame)
                # Envía la imagen por Telegram
                await send_image('imagen_detectada.jpg')

        # Muestra el video en tiempo real
        cv2.imshow('Camera - Custom Model Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Iniciar el bucle de eventos asyncio y ejecutar la función principal
    asyncio.run(main())
