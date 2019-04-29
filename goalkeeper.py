import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ap = argparse.ArgumentParser(description='Хоккейный вратарь. Отбивает шайбы по льду. При условии. что шайба движется по прямой.')
ap.add_argument("-c", "--coordinat", type = int, required = False, help="Вывод изображения на координатной оси.")
ap.add_argument("-p", "--predict", type = int, required = False, help="Предсказание для введенного числа X.")
ap.add_argument("-r", "--range", type = int, required = False, help="Определение параметров RGB.")
ap.add_argument("-b", "--ball", type = int, required = False, help="Поиск мяча.")
args = vars(ap.parse_args())

# Калибровка по цвету
def range(camera):

    def nothing(x):
        pass

    cap = cv2.VideoCapture(camera)
    cv2.namedWindow('result')

    cv2.createTrackbar('minb', 'result', 0, 255, nothing)
    cv2.createTrackbar('ming', 'result', 0, 255, nothing)
    cv2.createTrackbar('minr', 'result', 0, 255, nothing)

    cv2.createTrackbar('maxb', 'result', 0, 255, nothing)
    cv2.createTrackbar('maxg', 'result', 0, 255, nothing)
    cv2.createTrackbar('maxr', 'result', 0, 255, nothing)

    while(cap.isOpened()):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv', hsv)

        minb = cv2.getTrackbarPos('minb', 'result')
        ming = cv2.getTrackbarPos('ming', 'result')
        minr = cv2.getTrackbarPos('minr', 'result')

        maxb = cv2.getTrackbarPos('maxb', 'result')
        maxg = cv2.getTrackbarPos('maxg', 'result')
        maxr = cv2.getTrackbarPos('maxr', 'result')

        mask = cv2.inRange(hsv, (minb,ming,minr), (maxb,maxg, maxr))
        cv2.imshow('mask', mask)
        result = cv2.bitwise_and(frame, frame, mask = mask)
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Линейная регрессия
def predict(X):
    # прямая линия от (0,0) до (10,10)
    # преобразование x в двумерный массив, т.е. 1 колонка и необходимое количество рядов
    x = np.array([1, 2, 4, 5, 6, 8, 10]).reshape((-1, 1))
    y = np.array([1, 2, 4, 5, 6, 8, 10])

    print(x)
    print(y)

    # создание модели с параметрами по умолчанию
    # .fit() - вычисляются оптимальные значение весов 𝑏₀ и 𝑏₁
    model = LinearRegression().fit(x,y)

    # .score() принимает в качестве аргументов предсказатель x и регрессор y, и возвращает значение 𝑅².
    r_sq = model.score(x,y)
    print('coefficient of determination:', r_sq)

    # model содержит атрибуты .intercept_, который представляет собой коэффициент, и 𝑏₀ с .coef_, которые представляют 𝑏₁:
    # получение 𝑏₀ и 𝑏₁
    # .intercept_ – это скаляр, в то время как .coef_ – массив
    # примерное значение 𝑏₀ = -8.881784197001252e-16 показывает, что наша модель предсказывает ответ -8.881784197001252e-16 при 𝑥, равным нулю. Равенство 𝑏₁ = 1. означает, что предсказанный ответ возрастает до 1 при 𝑥, увеличенным на единицу.
    print('intercept (b0):', model.intercept_)
    print('slope (b1):', model.coef_)

    # предсказание
    # оценочная функция регрессии выражается уравнением 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥
    # y_pred = model.predict(x)
    # или
    # y_pred = model.intercept_ + model.coef_ * x

    # предсказание для x = 23
    #x0 = np.array([23]).reshape((-1, 1))
    x0 = np.array([X]).reshape((-1, 1))
    y_pred = model.predict(x0)
    print('При X =', int(x0))
    print('Предсказание для Y =', int(y_pred))

# Поиск мяча
def ball():
    print('Test')




if __name__ == '__main__':

    if args["coordinat"] is not None:
        cap = cv2.VideoCapture(args["coordinat"])

        _, frame = cap.read()

        plt.imshow(frame)
        plt.show()

    if args["predict"] is not None:
        predict(args["predict"])

    if args["range"] is not None:
        range(args["range"])

    if args["ball"] is not None:
        cap = cv2.VideoCapture(args["ball"])

        while(cap.isOpened()):
            _, frame = cap.read()

            cv2.imshow('Original', frame)
            ball()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
