import logging

import cv2
import serial

from img import IMGWorker

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

ser = serial.Serial('', 9600, timeout=1)

command = {
    'No line': 'STOP',
    'STRAIGHT': 'Go forward',
    'RIGHT': 'Go right',
    'LEFT': 'Go left'
}
last_direction = ''
counter = 0

while True:
    try:
        ret, frame = video.read()

        img_worker = IMGWorker(frame)
        output_img = img_worker.draw_white_objects()
        trajectory = img_worker.draw_trajectory()

        cv2.imshow('Finded contours', output_img)
        cv2.imshow('Trajectory', trajectory)

        direction = img_worker.analyze_trajectory()

        # if last_direction is None:
        #     last_direction = direction
        #     counter = 1
        #     logging.info(f'Текущее направление: {last_direction}')
        # elif direction == last_direction:
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= 5:
        #         last_direction = direction
        #         counter = 0
        logging.info(f'Направление изменилось на: {direction}')

        ser.write((command[direction]).encode())
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