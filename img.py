import cv2
import numpy as np


class IMGWorker:
    def __init__(self, frame: np.ndarray):
        """
        Args:
            - frame(numpy.ndarray) - цветное изображение в формате BGR
        """
        self.frame = frame


    def increase_img_brightness(self) -> np.ndarray:
        """
        Увеличивает яркость изображения для случаев, когда есть затемнения
        или когда просто недостаточное освещение

        Использует линейное преобразование яркости, т.е
        new_img =  alpha * img + beta,
        где alpha - коэффициент контрастности,
        а beta - значение яркости, которое мы добавляем к каждому пикселю

        Returns:
            - highlighted_img(numpy.ndarray) - изображение
                                               с увеличенной яркостью
        """
        highlighted_img = cv2.convertScaleAbs(self.frame, alpha=1.5, beta=70)
        return highlighted_img


    def get_white_mask(self) -> np.ndarray:
        """
        Выделяет синие объекты на изображении

        Механика:
            - Увеличиваем яркость изображения
            - Преобразовываем в HSV
            - Создаем маску для синего цвета
            - Удаляем шум и замыкания разрывов

        Returns:
            - clean_mask(numpy.ndarray) - бинарная маска
                                          с выделенными синими объектами
        """
        # img = self.increase_img_brightness()
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        return clean_mask


    def draw_white_objects(self) -> np.ndarray:
        """
        Отрисовывает контуры белых объектов на исходном изображении.
        При этом не учитываем объекты, площадь которых менее 500 пикселей,
        для исключения лишнего шума

        Returns:
            - output_img(numpy.ndarray) - исходное изображение с отрисованными
                                          контурами белых объектов
        """
        mask = self.get_white_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        output_img = self.frame.copy()
        cv2.drawContours(output_img, filtered_contours, -1,
                         (0, 0, 255), 2)
        return output_img

    def get_trajectory_points(self) -> list[tuple[int, int]]:
        """
        Получает траекторию в виде массива координат

        Returns:
            - points(list[tuple[int, int]]): массив координат
        """
        mask = self.get_white_mask()
        height, width = mask.shape
        points = []

        # Ищем индексы столбцов, у которых есть 255 (белый цвет)
        # если их несколько для одной строки, то берем их среднее значение
        for y in range(height):
            x = np.where(mask[y] == 255)[0]
            if len(x) > 0:
                mid_x = int(np.mean(x))
                points.append((mid_x, y))
        return points

    def draw_trajectory(self) -> np.ndarray:
        """
        Рисует траекторию белой линии на исходном изображении

        Returns:
            - output_img(numpy.ndarray) - изображенияе с отрисованной
                                          траекторией
        """
        points = self.get_trajectory_points()
        output_img = self.frame.copy()

        # Рисуем траекторию, соединяя соседние точки
        for i in range(1, len(points)):
            cv2.line(output_img, points[i-1], points[i],
                     (255, 0, 0), 2)

        return output_img


    def analyze_trajectory(self) -> str:
        """
        Анализирует траекторию и определяет ее отклонение влево/право

        Returns:
            str - направление отклонения траетории
        """
        points = self.get_trajectory_points()
        points_np = np.array(points)

        # Проверяем заполнен ли массив точками, т.е есть
        # ли в кадре белые объекты
        if not (points_np.size > 0):
            current_direction = "No line"
            return current_direction

        x = points_np[:, 0]
        y = points_np[:, 1]

        # Аппроксимируем значения методом наименьших квадратов
        # 1 в данном случае, степень полинома
        # возвращает [k, b] из y = kx + b
        fit = np.polyfit(y, x, 1)
        slope = fit[0]
        # Порог наклона для определения поворота
        threshold = 0.1

        if abs(slope) < threshold:
            current_direction = "RIGHT"
        elif slope > 0:
            current_direction = "LEFT"
        else:
            current_direction = "STRAIGHT"
        return current_direction
