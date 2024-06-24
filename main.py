#--------- Обучение и тестирование нейросети

# Подключение библиотеки для работы с многомерными массивами и высокоуровневыми математическими функциями
# NumPy - "Numerical Python"

import numpy as np

# Расчет функции активации (сигмоиды): f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# Расчет производной функции активации (сигмоиды): f'(x) = f(x) * (1 - f(x))
def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

# Расчет ошибки сети: y_true и y_pred - требуемое и расчетное значение выхода (выходов) сети
def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  Нейронная сеть со структурой:
    - 2 входами (i1, i2: i1- элемент массива x[0], i2 - элемент массива x[1])
    - 2 нейронами скрытого слоя (h1, h2)
    - 1 нейроном выходного слоя (o1)
  '''
  
  # Расчет начальных значений весов (w1-6) и смещений (b1-3)
  def __init__(self):
    # Расчет начальных значений весов (по нормальному случайному закону)
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Расчет начальных значений смещений (по нормальному случайному закону)
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()


  # РАБОТА СЕТИ
  # Расчет выходных значений нейронов h1, h2, o1 с использованием функции активации (сигмоидной функции)
  def feedforward(self, x):
    # x - массив значений входов сети
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  # Функция ОБУЧЕНИЯ СЕТИ
  def train(self, data, all_y_trues):
    
    '''
    - data - массив обучающих наборов данных (n x 2), n = к-во наборов в обучающей выборке, 2 - кол-во входов. 
    - all_y_trues - массив значений требуемых выходов для обучающих наборов (n).

      Элементы all_y_trues соответствуют элементам data.

    '''
    
    # Скорость обучения (греческая буква "эта": 0<"эта"<1)
    learn_rate = 0.7

    # Количество эпох (число обработок обучающей выборки)
    epochs = 1000 

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):

        # --- Расчет входных и выходных значений нейронов сети (будут использованы позже)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Расчет частных производных

        # --- Имена: d_L_d_w1 = "частная производная L по w1"

        d_L_d_ypred = -2 * (y_true - y_pred)

        # Нейрон o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Нейрон h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Нейрон h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Корректировка весов и смещений нейронов

        # Нейрон h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Нейрон h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Нейрон o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Расчет ошибки сети для каждой 10-й эпохи

      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
 
        # Вывод ошибки сети для каждой 10-й эпохи
        print("Эпоха %d Ошибка сети: %.3f" % (epoch, loss))
        print("W1= %.3f" % self.w1)
        print("W2= %.3f" % self.w2)
        print("W3= %.3f" % self.w3)
        print("W4= %.3f" % self.w4)
        print("W5= %.3f" % self.w5)
        print("W6= %.3f" % self.w6)

        print("b1= %.3f" % self.b1)
        print("b2= %.3f" % self.b2)
        print("b3= %.3f" % self.b3)
# Заполнение массивов обучающей выборки

# Значения входов для обучающих наборов 1-4
data = np.array([
  [700, 500],  # набор 1
  [550, 300],   # набор 2
  [300, 200],   # набор 3
  [150, 600], # набор 4
])

# Требуемые значения выхода для обучающих наборов 1-4
all_y_trues = np.array([
  1,         # набор 1
  1,         # набор 2
  1,         # набор 3
  1,         # набор 4
])

# ЗАПУСК ОБУЧЕНИЯ СЕТИ
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# ТЕСТИРОВАНИЕ СЕТИ
# Ввод тестовых наборов 1,2
test_n_1 = np.array([50, 1000]) # тестовый набор 1
test_n_2 = np.array([20, 2])  # тестовый набор 2

# Вывод результатов для тестовых наборов 1,2
print("Результат для тестового набора 1: %.3f" % network.feedforward(test_n_1)) # требуемое значение o1=1
print("Результат для Тестового набора 2: %.3f" % network.feedforward(test_n_2)) # требуемое значение o1=0
