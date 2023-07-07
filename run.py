import numpy as np
import cv2 as cv
from scipy.special import softmax
import time

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
    x[m_position[0] * width + m_position[1]] = 1
    x[t_position[0] * width + t_position[1]] = 1
    return minimize_weights(x)

def function(x): #функция активации
    #relu
    return np.maximum(x, 0)

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

f = open('cycle_weights.txt', 'r')

width = int(f.readline()) #ширина поля
first_layer = int(f.readline()) #ширина первого слоя
out_layer = int(f.readline()) #ширина выходного слоя

all_file = f.read() #считываем все до конца из файла как строку

f.close()

all_file = all_file.replace('[', '') #удаляем все '['
all_file = all_file.replace(']', '') #удаляем все ']'

#print(all_file)

lst_file = all_file.split() #превращаем строку в список

w = np.empty((width ** 2, first_layer)) #создаем матрицу весов первого слоя
wout = np.empty((first_layer, out_layer)) #создаем матрицу весов второго слоя

#заполняем матрицу весов первого слоя
for i in range(width ** 2): 
    lst_cur = lst_file[i * first_layer : (i + 1) * first_layer] #срез = строка матрицы
    for j in range(len(lst_cur)):
        w[i][j] = float(lst_cur[j]) #заполняем поэлементно

#удаляем из начала значения для матрицы весов первого слоя
lst_file = lst_file[(i + 1) * first_layer : ]

lst_cur = lst_file[ : first_layer] #срез = строка вектора b для первого слоя
b = []
for i in lst_cur:
    b.append(float(i)) #добавляем в b значения

#удаляем из начала значения для вектора b для первого слоя
lst_file = lst_file[first_layer : ]

#заполняем матрицу весов второго слоя
for i in range(first_layer):
    lst_cur = lst_file[i * out_layer : (i + 1) * out_layer] #срез = строка матрицы
    for j in range(len(lst_cur)):
        wout[i][j] = float(lst_cur[j]) #заполняем поэлементно

#удаляем из начала значения для матрицы весов второго слоя
lst_file = lst_file[(i + 1)* out_layer : ]

lst_cur = lst_file #оставшееся = строка вектора bout для выходного слоя
bout = []
for i in lst_cur:
    bout.append(float(i)) #добавляем в bout значения


t_position = [0, 0] #позиция цели
#m_position = [8, 8] #начальная позиция манипулятора
print('Enter ''x y'' of start position between 0 and ' + str(width))
m_position = [int(i) for i in input().split()] #начальная позиция манипулятора

#проверка правильности данных
if out_of_bounds(m_position) == 1:
    print('Wrong data')
    exit()

img = cv.imread("black_square.png", 0) #будущее изображение
img[m_position[0], m_position[1]] = 255
img[t_position[0], t_position[1]] = 255

dist = distance(m_position, t_position) #расстояние

time_before = time.time()

#строим путь к цели
while dist > 0 and time.time() < time_before + 1:
#пока цель не найдена и время выполнения <= 1

    x = sight(m_position, t_position) #преобразуем координаты в векторное представление
    t = x.T @ w + b #линейное преобразование
    h = function(t) #применяем нелинейность
    y = h @ wout + bout #получаем вектор ответов:
#y[i] = вероятность того, что перемещение номер i приблизит к цели
    
    change_position = action(y, m_position) #выбрали направление перемещения
    m_position = [m_position[0]  + change_position[0], m_position[1]  + change_position[1]]
    dist = distance(m_position, t_position)
    if dist > 0:
        img[m_position[0], m_position[1]] = 110 #отмечаем шаг на рисунке пути

img = cv.resize(img, (200, 200)) #рисунок пути
cv.imshow('way', img)

"""print(w)
print('\n\n\n')
print(b)
print('\n\n\n')
print(wout)
print('\n\n\n')
print(bout)
print('\n\n\n')"""
