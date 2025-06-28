import logging

import cv2
import serial

from img import IMGWorker

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01)

while True:
    try:
        ret, frame = video.read()

        if not ret or frame is None:
            logging.warning("Пустой кадр")
            continue

        img_worker = IMGWorker(frame)
        output_img = img_worker.draw_white_objects()
        trajectory = img_worker.draw_trajectory()

        cv2.imshow('Finded contours', output_img)
        cv2.imshow('Trajectory', trajectory)

        direction = img_worker.analyze_trajectory()

        logging.info(f'Направление: {direction}')

        if direction == 'No line':
            direction = 'STOP'

        ser.write(direction.encode())
        logging.info(f'Отправили команду {direction}')
        response = ser.readline().decode().strip()
        logging.info(f'Ответ ардуино: {response}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        logging.error(f'Ошибка: {e}')

ser.close()
video.release()
cv2.destroyAllWindows()