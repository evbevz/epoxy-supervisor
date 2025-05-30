from lxml import etree
import matplotlib.pyplot as plt
import os.path
import torch
from ultralytics import YOLO

# Косинусное сходство (Косинус угла между векторами)
def cos_sim (v1, v2):
    cosine_similarity = torch.dot(v1, v2) / (torch.linalg.vector_norm(v1) * torch.linalg.vector_norm(v2))
    return cosine_similarity

# Проекция вектора на вектор
def pr_vec (v1, v2):
    pr_v1_on_v2 = torch.dot(v1, v2) / torch.linalg.vector_norm(v2)
    return pr_v1_on_v2

# Загрузка файла анотации в формате "CVAT for images 1.1"
# Из файла берётся массив точек полилинии с разметкой шкалы деления: 0мл, 1мл, ... 20мл
def Calibrate (fAnnotation):
    tree = etree.parse(fAnnotation)
    root = tree.getroot()
    #result = root.xpath('//polyline[@label="injector meter"]')
    result = root.xpath('//polyline[@label="centroid axis"]')
    if not(result):
        raise Exception("Не удалось найти polyline с точками уровней инжектора в файле: ", fAnnotation)
    points = result[0].attrib['points']
    # Сложная конструкция, но по сути разбирает стороку на список списков (двумерный список) из двух значений попутно переводя из строковых значений в числовое значние.
    list_of_points = list(map(lambda x: list(map(float,x.split(','))), points.split(';')))
    array_of_points = torch.tensor(list_of_points)
    #print ("Массив точек шкалы инжектора:\n", array_of_points)
    return array_of_points

# Получение центра эллипса по точкам: левой и правой большой оси, дальней и ближней малой оси.
# Вычисляем через середину диагоналей эллипса. На будущее стоит просто определять bounding box этого эллипса. Модель будет выдавать уже как раз середину.
def GetEllipseCenter (kpt, kptConfidence):
    #print('--- GetEllipseCenter Keypoints: ---\n', kpt)
    if len(kpt) == 0:
        return None

    # Проверяем уверенность в определении точки. Если показатель меньше заданного значения, то точку игнорируем.
    if (kpt[0][2] > kptConfidence) :
        ellipse_center = kpt[0][0:2:]
    else:
        ellipse_center = None
    return ellipse_center

def get_level_by_kpt (keypoints, kptConfidence, arrayEpoxyLevel):
    #print('--- Keypoints instance: ---\n', keypoints)
    # Высчитываем центр эллипса
    # Передаем в параметре 4 точки диагоналей эллипса в виде тензора 4x3 [[x1,y2,confidence1],[]...] и коэффициент уверенности в правильности распознавания, ниже которого не будем считать, что точки определились правильно. Т.е. координаты такой точки будем считать ложными и точку игнорировать.
    #ellipse_center = GetEllipseCenter(keypoints.data[0][0:4:], kptConfidence)
    # Модель распознает четыре точки краев шприца и последнюю пятую точку центра эллипса/уровня эпоксидки. Поэтому передаем только точку центра
    ellipse_center = GetEllipseCenter(keypoints.data[0][4:5:], kptConfidence)

    if ellipse_center is not None:
        # Переносим массив точек шприца на то же устройство рассчета где и тензоры модели предсказаний. Если расчёты велись на CUDA, то лучше там и считать всё остальное.
        device = ellipse_center.device
        arrayEpoxyLevel = arrayEpoxyLevel.to(device)

        # Вычисляем ближайшую калиброванную точку к предсказанной точке (середине эллипса)
        LengthMin = (keypoints.orig_shape[0] ** 2 + keypoints.orig_shape[1] ** 2) ** 0.5 # Нужно просто большое значение, но решил указать максимально возможное расстояние на изображении (диагональ)
        LevelMin = 0
        for level, kpt in enumerate(arrayEpoxyLevel):
            Length = torch.norm(ellipse_center - kpt) # Расстояние между предсказанной точкой и калиброванной точкой уровня
            if Length < LengthMin:                    # Если решим, что хоти чтобы при одинаковом расстоянии показывал значение большего уровня, то тогда поставить знак сравнения <=
                LengthMin = Length                    # Запоминаем минимальное расстояние
                LevelMin = level                      # Запоминаем уровень к точке которого расстояние минимальное
            #print ("Lvl: ", level, "\tCalib pt: ", kpt, "\tPredict pt: ", ellipse_center, "\tLength: ", Length)
        #print ("Предсказанный уровень эпоксидки: ", LevelMin)

        # Дорасчёт дробной части уровня.
        # Определяем с какой стороны от ближайшей калиброванной точки уровня находится точка предсказания и на каком расстоянии
        vSrc = ellipse_center - arrayEpoxyLevel[LevelMin]
        if (LevelMin == (arrayEpoxyLevel.size(dim=0) - 1)): # Ограничение на верхний уровень инжектора
            vNext = torch.tensor([0.0, 0.0]).to(device)
        else:
            vNext = arrayEpoxyLevel[LevelMin+1] - arrayEpoxyLevel[LevelMin]
        if (LevelMin == 0): # Ограничение на нижний уровень инжектора
            vPrev = torch.tensor([0.0, 0.0]).to(device)
        else:
            vPrev = arrayEpoxyLevel[LevelMin-1] - arrayEpoxyLevel[LevelMin]

        cos_next = cos_sim(vSrc,vNext)
        cos_prev = cos_sim(vSrc,vPrev)

        #print ("Lvl: ", LevelMin+1, "\tXY next: ", arrayEpoxyLevel[LevelMin+1], "\tvNext: ", vNext, "\tL2: ", torch.linalg.vector_norm(vNext), "\tCos2Next: ", cos_next)
        #print ("Lvl: ", LevelMin, "\tXY el_c: ", arrayEpoxyLevel[LevelMin], "\tvSrc: ", vSrc, "\tL2: ", torch.linalg.vector_norm(vSrc))
        #print ("Lvl: ", LevelMin-1, "\tXY prev: ", arrayEpoxyLevel[LevelMin-1], "\tvPrev: ", vPrev, "\tL2: ", torch.linalg.vector_norm(vPrev), "\tCos2Prev: ", cos_prev)

        # Если точка выше ближайшего калиброванного уровня, то добавляем дорасчитанную дробную часть объема
        if cos_next > cos_prev:
            pr = pr_vec(vSrc, vNext)
            pr_ml = float (1 / torch.linalg.vector_norm(vNext) * pr)
        # Если точка ниже ближайшего калиброванного уровня, то вычитаем дорасчитанную дробную часть объема
        elif cos_next < cos_prev:
            pr = pr_vec(vSrc, vPrev)
            pr_ml = - float (1 / torch.linalg.vector_norm(vPrev) * pr)
        else:
            pr = 0
            pr_ml = 0
        #print("Длина проекции в пикселях: ", pr, "\tДлина проекции в мл: ", pr_ml)
        return LevelMin + pr_ml
    else:
        #print("Ellipse_center is None.")
        return None

# Получение уровня эпоксидки по картинке
def GetEpoxyLevel (model, arrayEpoxyLevel, filenameInjectorCam, kptConfidence, level_prev):
    # Запускаем предсказание
    results = model.predict(source=filenameInjectorCam, verbose=False, save=False)  # Предсказание по изображению. Возвращается список результатов (т.к. можно передать список кадров или даже видео)
    #print ('--- Results[0].Boxes: ---\n', results[0].boxes)
    #print ('--- Results[0].Keypoints: ---\n', results[0].keypoints)
    # Теоретически может быть список результатов, но берём только одно - первое изображение для распознавания и определения уровня.
    keypoints = results[0].keypoints  # Keypoints object for pose outputs
    #results[0].show()  # display to screen
    #print (f'--- Keypoints: ---\n {keypoints}')

    # Если предсказание ничего не нашло на фото, то возвращаем предыдущий уровень
    if keypoints.conf is None:
        #print(f"Уровень не распознан. Прошлый уровень эпоксидки: {level_prev}")
        return level_prev
        
    length_min = 21
    # Если предыдущий уровень не определен, то скорее всего это первый вызов и тогда предполагаем, что инжектор заправлен максимально (20мл) и выбирать надо ближайший к максимальной заправке.
    if level_prev is None:
        level_prev = 20
    # Из всех распознанных уровней эпоксидки выбираем ближайший по расстоянию к предыдущему уровню
    level_near_prev = None
    for kpts_instance in keypoints:
        level_cur = get_level_by_kpt(kpts_instance, kptConfidence, arrayEpoxyLevel)
        if level_cur is None:
            #print("level_cur is None")
            break
        length = abs(level_prev - level_cur)
        if length < length_min:                         # Если расстояние между одним из предсказанных уровней и предыдущим наименьшее, то 
            length_min = length                         # Запоминаем минимальное расстояние
            level_near_prev = level_cur                 # Запоминаем уровень ближайший к предыдущему
        #print (f"Previous lvl: {level_prev},\tCurrent lvl: {level_cur},\tLength between: {length}\tMin length: {length_min}")
    #print (f"Ближайший уровень эпоксидки: {level_near_prev}")
    return level_near_prev    

