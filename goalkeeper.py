import numpy as np
from sklearn.linear_model import LinearRegression

# Линейная регрессия
def predict():
    # прямая линия от (0,0) до (10,10)
    # преобразование x в двумерный массив, т.е. 1 колонка и необходимое количество рядов
    x = np.array([0, 1, 2, 4, 5, 6, 8, 10]).reshape((-1, 1))
    y = np.array([0, 1, 2, 4, 5, 6, 8, 10])

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
    x0 = np.array([23]).reshape((-1, 1))
    y_pred = model.predict(x0)
    print('if x = ', int(x0))
    print('predicted response:', int(y_pred), sep='\n')

if __name__ == '__main__':
    predict()
