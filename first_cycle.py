import cv2 as cv
import numpy as np
import random
from scipy.special import softmax
from scipy.stats import uniform
import matplotlib.pyplot as plt
import time
import math

#гиперпараметры

width = 9 #ширина рабочего стола
first_layer = int(8 * width ** 2)
out_layer = 4
alpha = 0.01
train = 10 #количество циклов обучения

def distance(point1, point2):
    #point - вектор [x, y]
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def minimize_weights(m):
    #m - матрица
    s = np.sum(m)
    if abs(s) < 1:
        return(m)
    m = m / s
    return m

def sight(m_position, t_position): #взгляд на рабочую поверхность в виде списка
    x = np.zeros((width ** 2, 1))
    x[m_position[0] * width + m_position[1]] = 255
    x[t_position[0] * width + t_position[1]] = 255
    return minimize_weights(x)

def function(x): #функция активации
    #relu
    return np.maximum(x, 0)
#    for i in range(len(x)):
#        for j in range(len(x[i])):
#            x[i][j] = 0.5 + math.atan(x[i][j]) / math.pi
#    return x

def function_deriv(t):
    #relu derivation
    return (t >= 0).astype(float)

def out_of_bounds(position): #проверка на выход за пределы рабочей поверхности
    if position[0] >= width or position[0] < 0:
        return 1
    if position[1] >= width or position[1] < 0:
        return 1
    return 0

def action(x, m_position): #одно перемещение манипулятора
    x = softmax(x[0])
    max_n = 0
    for i in range(1, len(x)):
        if x[i] >= x[max_n]:
            max_n = i
    possible_actions = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
    delta_position = possible_actions[max_n] #изменение позиции манипулятора
    if not out_of_bounds([m_position[0] + delta_position[0], m_position[1] + delta_position[1]]):
        return delta_position
    else:
        return [0, 0]
    
def error(z, y):
    #sparse_cross_entropy
    for i in range(len(z[0])):
        if y[0][i] == 0:
            y[0][i] = 0.0000001
    return -np.sum(np.log(y[0]) * z[0])

def optimal_way(m_position, t_position): #лучшее действие для манипулятора на данном шаге
    #possible_actions = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
    opt_y = np.zeros((out_layer, 1))
    if m_position[0] < t_position[0]:
        opt_y[0] = 1
        return opt_y
    if m_position[0] > t_position[0]:
        opt_y[2] = 1
        return opt_y
    if m_position[0] < t_position[0]:
        opt_y[1] = 1
        return opt_y
    if m_position[0] > t_position[0]:
        opt_y[3] = 1
        return opt_y
    return opt_y

#основная часть

#генерация списка возможных позиций цели и манипулятора
m_positions = [] #манипулятор
t_positions = [[0, 0]] #цель
for i in range(width):
    for j in range(width):
        m_positions.append([i, j])
#        t_positions.append([i, j])
#m_positions = np.random.permutation(m_positions) #перемешиваем для лучшей обучаемости
#t_positions = np.random.permutation(t_positions) #перемешиваем для лучшей обучаемости

loss_arr = [] #список значений лоссов для графика

w = np.random.rand(width ** 2, first_layer) #для первого слоя
#b = np.random.rand(first_layer) #для первого слоя
wout = np.random.rand(first_layer, out_layer) #для второго(выходного) слоя
#bout = np.random.rand(out_layer) #для второго(выходного) слоя

for i in range(len(w)):
    w[i] = np.random.uniform(-1, 1, first_layer)
b = np.random.uniform(-1, 1, first_layer)
for i in range(len(wout)):
    wout[i] = np.random.uniform(-1, 1, out_layer)
bout = np.random.uniform(-1, 1, out_layer)

#print(time.time()) #при желании смотрим время рассчетов

for t in range(train):
    m_positions = np.random.permutation(m_positions) #перемешиваем для лучшей обучаемости
    for cycle_i in range(len(m_positions)):
        for cycle_j in range(len(t_positions)):

            m_position = m_positions[cycle_i] #начальная позиция манипулятора
            t_position = t_positions[cycle_j] #позиция цели

#        print(m_position, t_position)

            dist = distance(m_position, t_position)

            while dist > 0: #пока не дошли до цели

                y_opt = optimal_way(m_position, t_position).T #лучшее возможное действие
                x = sight(m_position, t_position) #преобразуем координаты в векторное представление
                t = x.T @ w + b #линейное преобразование
                h = function(t) #применяем нелинейность

#            if np.sum(h @ wout) > 10 ** 30:
#            print(h, wout, bout)
#                break
            
                y = h @ wout + bout #получаем вектор ответов:
#y[i] = вероятность того, что перемещение номер i приблизит к цели
            
                change_position = action(y, m_position) #выбрали направление перемещения
                new_position = [m_position[0]  + change_position[0], m_position[1]  + change_position[1]]
                new_dist = distance(new_position, t_position)
            
                if new_dist >= dist: #из-за выбранного перемещения остались на месте или
#отдалились от цели

                    loss = 1
                    while loss > 0.01:
                        #обратное распространение ошибки
                        dloss_dy = softmax(y) - y_opt
                        dloss_dwout = h.T @ dloss_dy
                        dloss_dbout = dloss_dy
                        dloss_dh = dloss_dy @ wout.T
                        dloss_dt = dloss_dh * function_deriv(t)
                        dloss_dw = x @ dloss_dt
                        dloss_db = dloss_dt

                        #изменение весов
                        w = w - alpha * dloss_dw
                        b = b - alpha * dloss_db
                        wout = wout - alpha * dloss_dwout
                        bout = bout - alpha * dloss_dbout

                        #пересчет лосса
                        t = x.T @ w + b
                        h = function(t)
                        y = h @ wout + bout
                        loss = error(y_opt, softmax(y))
                        loss_arr.append(loss) #запоминаем лосс для построения графика при желании
    
                else: #новая позиция ближе к цели
                    m_position = new_position #меняем текущую позицию на новую
                    dist = new_dist
                    loss = error(y_opt, softmax(y))
                    loss_arr.append(loss) #запоминаем лосс для построения графика при желании

#print(time.time()) #при желании смотрим время рассчетов

#print(y, wout, bout, "!!!", h, w, b)

f = open('cycle_weights.txt', 'w')
f.write(str(width) + '\n')
f.write(str(first_layer) + '\n')
f.write(str(out_layer) + '\n')
for i in w:
    f.write(str(i) + ' ')
for i in b:
    f.write(str(i) + ' ')
for i in wout:
    f.write(str(i) + ' ')
for i in bout:
    f.write(str(i) + ' ')
f.close()

#при желании строим график лоссов
#plt.plot(loss_arr)
#plt.show()


#print(w)
#print(b)
#print(wout)
#print(bout)


"""
+ограничение движения
+изменить action под out_layer выходных нейронов
+изменить размерность матриц под размерности внутреннего и внешнего слоя
+изменение матриц в обратном спуске
+уменьшение весов
-все в цикл
-отрисовка
-запуск
"""
