import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from collections import Counter
from keras.utils.data_utils import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from pyjarowinkler import distance as jaro_winkler_distance
import re
import string as s
import random
import openpyxl

# Класс Product, который имеет свойства Имя, Цена, Доля
class Product:
    def __init__(self, name):
        self.name = name
        self.weight = None
        self.priceBYN = None
        self.priceRUB = None
        self.parrent = None
    
    def SetParrent(self, parrent):
        self.parrent = parrent
    
    def SetWeight(self, weight):
        self.weight = weight
    def SetPriceBYN(self, price):
        self.priceBYN = price
    
    def SetPriceRUB(self, price):
        self.priceRUB = price

# Класс Subgroup, который имеет Имя, список объектов Product, метод, который возвращает список объектов Product
class Subgroup:
    def __init__(self, name):
        self.name = name
        self.products = [] # список объектов Product

    # Метод, который возвращает список объектов Product
    def GetProducts(self):
        return self.products

    # Функция Append, которая добавляет объект Product в список products
    def Append(self, product):
        if isinstance(product, Product): # проверяем, что аргумент является объектом Product
            product.SetParrent(self)
            self.products.append(product) # добавляем его в список products
        else:
            print("Invalid argument") # иначе выводим сообщение об ошибке

# Класс Group, который имеет Имя, список объектов Product, список объектов Subgroup,
# имеет метод, который возвращает список объектов Product, который содержит сам и который содержится в объектах Subgroup
class Group:
    def __init__(self, name):
        self.name = name
        self.products = [] # список объектов Product
        self.subgroups = [] # список объектов Subgroup

    # Метод, который возвращает список объектов Product, который содержит сам и который содержится в объектах Subgroup
    def GetProducts(self):
        result = [] # пустой список для результата
        result.extend(self.products) # добавляем в него свои продукты
        for subgroup in self.subgroups: # проходим по подгруппам
            result.extend(subgroup.GetProducts()) # добавляем в него продукты из подгрупп
        return result

    # Функция Append, которая добавляет объект Product или Subgroup в соответствующий список
    def Append(self, item):
        if isinstance(item, Product): # проверяем, что аргумент является объектом Product
            item.SetParrent(self)
            self.products.append(item) # добавляем его в список products
        elif isinstance(item, Subgroup): # проверяем, что аргумент является объектом Subgroup
            self.subgroups.append(item) # добавляем его в список subgroups
        else:
            print("Invalid argument") # иначе выводим сообщение об ошибке

# Класс Category, который имеет Имя и метод получения списка всех объектов Product у всех дочерних объектов
class Category:
    def __init__(self, name):
        self.name = name
        self.groups = [] # список объектов Group

    # Метод получения списка всех объектов Product у всех дочерних объектов
    def GetProducts(self):
        result = [] # пустой список для результата
        for group in self.groups: # проходим по группам
            result.extend(group.GetProducts()) # добавляем в него продукты из групп и подгрупп
        return result

    # Функция Append, которая добавляет объект Group в список groups
    def Append(self, group):
        if isinstance(group, Group): # проверяем, что аргумент является объектом Group
            self.groups.append(group) # добавляем его в список groups
        else:
            print("Invalid argument") # иначе выводим сообщение об ошибке

def loadProdPricesRB(path='RB.xls', header=7):
    ''' Загрузить продовольственные товары РБ из файла с ценами
    '''
    df = pd.read_excel(path, header=header)
    prod = df.iloc[:df[df['ПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'] == 'НЕПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'].index[0]]
    prod = prod.dropna()
    return prod

def loadNotProdPricesRB(path='RB.xls', header=7):
    ''' Загрузить непродовольственные товары РБ из файла с ценами
    '''
    df = pd.read_excel(path, header=header)
    notprod = df.iloc[df[df['ПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'] == 'НЕПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'].index[0]+1:]
    notprod = notprod.dropna()
    return notprod

def clean_string(string):
  '''Функция очистки
  Получает строку, переводит в нижний регистр, очищает лишних данных, пробелов, знаков препинания и т.д.
  '''
  string = string.lower()
  # Ищем и заменяем скобки и все, что в них, на пустую строку
  string = re.sub(r"\(.*\)", " ", string)

  # Удаляем знаки препинания
  string = string.translate(str.maketrans(' ', ' ', s.punctuation))

  # Удаляем единицы измерения
  pattern = r'(\d+)*\s+(кг|г|л|мл|шт|штук|м|м2|м3|см|таблеток|пакетиков)\b' # шаблон регулярного выражения
  string = re.sub(pattern, " ", string) # замена единиц измерения на пустую строку
  # Удаляем цифры
  string = re.sub(r'\d+', " ", string)
  # Удаляем лишние пробелы
  string = re.sub(r"\s+", " ", string)
  
  # Удаляем лишние пробелы в начале и конце строки
  string = string.strip()

  # Добавляем очищенную строку в список
  return string

def clean(df, column):
  '''Применяет функцию clean_string() для всего столбца и записывает новые строки в новый столбец с припиской _cleaned'''
  df[column+'_cleaned'] = df[column].apply(lambda x: clean_string(x))
  return df

def my_levenshtein_distance(str1, str2):
   '''Функция для получения разницы между строками методом Левенштейна (не тот мужик из американского пирога),
     при котором строка разбивается на слова и сортируется по алфавиту
   '''
   str1_words = str1.split()
   str2_words = str2.split()

   str1_words.sort()
   str2_words.sort()

   str1_sorted = ' '.join(str1_words)
   str2_sorted = ' '.join(str2_words)

   return levenshtein_distance(str1_sorted, str2_sorted)

def combine_cosine_levenshtein_jar_win(str1, str2, alpha=0.25, beta=0.25, gamma=0.5):
    '''Функция сопоставления двух строк str1 и str2. Возвращает значение от 0 до 1, насколько похожи строки.
    Функция использует комбинацию из методов косинусного сопоставления, Левенштейна и Джаро-Винклера\n
    alpha: вес для метода косинусного сопоставления\n
    beta: вес для метода Левенштейна
    gamma: вес для метода Джаро-Винклера'''
    # convert strings to vectors of character frequencies
    vec1 = list(Counter(str1).values())
    vec2 = list(Counter(str2).values())
    # pad the vectors with zeros to the same length
    vec1, vec2 = pad_sequences([vec1, vec2], padding='post')
    # compute cosine similarity between vectors
    cos_sim = cosine_similarity([vec1], [vec2])[0][0]
    # normalize cosine similarity to [0, 1] range
    cos_sim = (1 + cos_sim) / 2
    # compute levenshtein distance between strings
    lev_dist = my_levenshtein_distance(str1, str2)
    # normalize levenshtein distance to [0, 1] range
    lev_dist = 1 - lev_dist / max(len(str1), len(str2))
    # compute jaro-winkler distance between strings
    jar_win_dist = jaro_winkler_distance.get_jaro_distance(str1, str2)
    # combine the three methods by weighted sum
    return alpha * cos_sim + beta * lev_dist + gamma * jar_win_dist

def compareRB(list1, df2, colname2, colname2prices, perc, alpha, beta, gamma):
    list2 = df2[colname2].to_list()
    resultdf = pd.DataFrame()
    for i in range(len(list1)):
        name1 = list1[i].name
        min_dist = 0
        for j in range(len(list2)):
            score = combine_cosine_levenshtein_jar_win(name1, list2[j], alpha, beta, gamma)
            if score > min_dist:
                min_dist = score
                similar_index = j
        if min_dist < perc:
            if isinstance(list1[i].parrent, Subgroup):
                name1 = list1[i].name + ' ' + list1[i].parrent.name

        if min_dist >= perc:
            list1[i].SetPriceBYN(df2[colname2prices].iloc[similar_index])
            resultdf = resultdf.append({'name1': name1,
                                        'name2': list2[similar_index],
                                        'score': min_dist,
                                        'weight': list1[i].weight,
                                        'priceBYN': list1[i].priceBYN}, ignore_index=True)
        else:
            if name1 == list1[i].name:
                if isinstance(list1[i].parrent, Subgroup):
                    name1 = list1[i].name + ' ' + list1[i].parrent.name
                i = i - 1
            else:
                resultdf = resultdf.append({'name1': list1[i].name}, ignore_index=True)
    return resultdf

def compareRF(list1, df2, colname2, colname2prices, perc, alpha, beta, gamma):
    list2 = df2[colname2].to_list()
    resultdf = pd.DataFrame()
    for i in range(len(list1)):
        name1 = list1[i].name
        min_dist = 0
        for j in range(len(list2)):
            score = combine_cosine_levenshtein_jar_win(name1, list2[j], alpha, beta, gamma)
            if score > min_dist:
                min_dist = score
                similar_index = j
        if min_dist >= perc:
            print(df2[colname2prices].iloc[similar_index])
            list1[i].SetPriceRUB(df2[colname2prices].iloc[similar_index])
            resultdf = resultdf.append({'name1': name1,
                                        'name2': list2[similar_index],
                                        'score': min_dist,
                                        'weight': list1[i].weight,
                                        'priceRUB': list1[i].priceRUB}, ignore_index=True)
        else:
            if name1 == list1[i].name:
                if isinstance(list1[i].parrent, Subgroup):
                    name1 = list1[i].name + ' ' + list1[i].parrent.name
                i = i - 1
            else:
                resultdf = resultdf.append({'name1': list1[i].name}, ignore_index=True)
    return resultdf

def GetDisparity(price1, price2, rate, weight1=1, weight2=1):
    '''
    price1 - цена товара в первой валюте\n
    price2 - цена товара во второй валюте\n
    rate - курс валюты\n
    weight1 - вес товара в первой корзине\n
    weight2 - вес товара во второй корзине
    '''

    return (price1 * weight1 / rate) / (price2 * weight2)

def random_weights():
    '''Возвращает случайные веса для трёх методов
    '''
    alpha = random.random()
    beta = random.random()
    gamma = random.random()
    total = alpha + beta + gamma
    alpha = round(alpha / total, 2)
    beta = round(beta / total, 2)
    gamma = round(gamma / total, 2)
    return alpha, beta, gamma

def compare_rb_rf(list1, list2, perc, alpha = 0.25, beta = 0.5, gamma = 0.25, noProd = False):
    '''Функция нахождения максимально похожих строк, используя цикл и случайные веса
    '''
    # Создаем пустой датафрейм
    resultdf = pd.DataFrame()
    # Используем цикл сопоставления
    for i in range(len(list1)):
        min_dist = 0
        name1 = clean_string(list1[i].name)
        if noProd:
            name1 = clean_string(list1[i].name + ' ' + list1[i].parrent.name)
        for j in range(len(list2)):
            name2 = clean_string(list2[j].name)
            score = combine_cosine_levenshtein_jar_win(name1, name2, alpha, beta, gamma)
            if score > min_dist:
                min_dist = score
                similar_index = j
        if min_dist >= perc and list1[i].priceBYN != None:
            priceRUB = list2[similar_index].priceRUB
            if priceRUB != None:
                list1[i].SetPriceRUB(priceRUB)
                disparity = GetDisparity(list1[i].priceBYN, priceRUB, 0.035618) # Стоимость рубля временная. Добавить функцию получения автоматически
                # Записываем в итоговый датафрейм
                resultdf = resultdf.append({'name1': list1[i].name,
                                'price_rb': list1[i].priceBYN,
                                'name2': list2[similar_index].name,
                                'price_rf': priceRUB,
                                'disparity': disparity,
                                'score': min_dist}, ignore_index=True)
            else:
                resultdf = resultdf.append({'name1': list1[i].name,
                                'price_rb': list1[i].priceBYN,
                                'name2': list2[similar_index].name,
                                'price_rf': '',
                                'disparity': '',
                                'score': ''}, ignore_index=True)
        else:
            resultdf = resultdf.append({'name1': list1[i].name,
                            'price_rb': list1[i].priceBYN,
                            'name2': '',
                            'price_rf': '',
                            'disparity': '',
                            'score': ''}, ignore_index=True)
    return resultdf

def FindIndexByText(ws, column, text):
    for i in range(9, ws.max_row + 1):
        # Получаем значение ячейки в первом столбце
        value = ws.cell(i, column).value
        # Сравниваем значение с искомым текстом
        if value != None and text.lower() in value.lower():
            # Выводим индекс строки
            return i

df = pd.read_excel("Average_prices-06-2023.xls", header=6)
df = df.rename(columns={df.columns[0]: 'Название товара'})

prodPricesStartIndex = df[df['Название товара'].str.lower() == 'продовольственные товары'].index[0]
prodPricesEndIndex = df[df['Название товара'].str.lower() == 'непродовольственные товары'].index[0]
dfProdPrices = df[prodPricesStartIndex+1:prodPricesEndIndex]
# Удаляет все строки, где хоть в одной ячейке есть nan. Заменить на удаление только тех строк, где вообще все ячейки пустые
dfProdPrices = dfProdPrices.dropna()

noProdPricesStartIndex = df[df['Название товара'].str.lower() == 'непродовольственные товары'].index[0]
dfNoProdPrices = df[noProdPricesStartIndex+1:]
# Удаляет все строки, где хоть в одной ячейке есть nan. Заменить на удаление только тех строк, где вообще все ячейки пустые
dfNoProdPrices = dfNoProdPrices.dropna()

df2 = pd.read_excel("! Копия Веса.xlsx", sheet_name='РБ_Струкутура', header=8)
dfProdWeights = df2.iloc[:df2[df2['ПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ (без табачных изделий)'] == 'НЕПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'].index[0]]
df2 = pd.read_excel("! Копия Веса.xlsx", sheet_name='РБ_Струкутура', header=8)
# df = df.rename(columns={'ПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ (без табачных изделий)':'name'})
dfNoProdWeights = df2.iloc[df2[df2['ПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ (без табачных изделий)'] == 'НЕПРОДОВОЛЬСТВЕННЫЕ ТОВАРЫ'].index[0]+1:]

wb = openpyxl.load_workbook("! Копия Веса.xlsx")
ws = wb.get_sheet_by_name('РБ_Струкутура')
        
prodIndex = FindIndexByText(ws, 1, 'Продовольственные товары')
noProdIndex = FindIndexByText(ws, 1, 'Непродовольственные товары')
serviceIndex = FindIndexByText(ws, 1, 'Услуги')

prodCategoryRB = Category('Продовольственные товары')

for i in range(prodIndex+1, noProdIndex):
    cell = ws.cell(i, 1)
    weightCell = ws.cell(i,2)
    value = cell.value
    weight = weightCell.value
    font = cell.font
    if weight:
        if font.bold and not font.italic and not font.underline:
            print('Group: ',value)
            group = Group(value)
            prodCategoryRB.Append(group)
            lastList = group
        elif not font.bold and font.italic and font.underline:
            print('Subgroup', value)
            subgroup = Subgroup(value)
            group.Append(subgroup)
            lastList = subgroup
        elif not font.bold and not font.italic and not font.underline:
            print('Product:', value, '-', weight)
            product = Product(value)
            product.SetWeight(weight)
            lastList.Append(product)
        
noProdCategoryRB = Category('Непродовольственные товары')

for i in range(noProdIndex+1, serviceIndex):
    cell = ws.cell(i, 1)
    value = cell.value
    weightCell = ws.cell(i, 2)
    weight = weightCell.value
    font = cell.font
    if weight:
        if font.bold and not font.italic and not font.underline:
            print('Group: ',value)
            group = Group(value)
            noProdCategoryRB.Append(group)
            lastList = group
        elif font.bold and font.italic:
            print('SubGroup', value)
            subgroup = Subgroup(value)
            group.Append(subgroup)
            lastList = subgroup
        elif not font.bold and not font.italic and not font.underline:
            print('Product:', value, '-', weight)
            product = Product(value)
            product.SetWeight(weight)
            lastList.Append(product)

prodPricesRB = loadProdPricesRB()
prodPricesRB = clean(prodPricesRB, prodPricesRB.columns[0])
notProdPricesRB = loadNotProdPricesRB()
notProdPricesRB = clean(notProdPricesRB, notProdPricesRB.columns[0])

prodRB = prodCategoryRB.GetProducts()
noProdRB = noProdCategoryRB.GetProducts()

comparedProdRB = compareRB(prodRB, prodPricesRB, prodPricesRB.columns[-1], prodPricesRB.columns[1],
                        0.75, alpha=0.25, beta=0.5, gamma=0.25)

comparednoProdRB = compareRB(noProdRB, notProdPricesRB, notProdPricesRB.columns[-1], notProdPricesRB.columns[1],
                        0.75, alpha=0.25, beta=0.5, gamma=0.25)

# Загружаем продовольственные и непродовольственные товары РФ
rf = pd.read_excel('! Копия Веса.xlsx', sheet_name='РФ_структура', header=3)
rf = rf.drop(axis=1, columns=[rf.columns[2], rf.columns[3], rf.columns[4], rf.columns[5], rf.columns[6], rf.columns[7], rf.columns[8], rf.columns[9], rf.columns[10]])

prodIndexRF = rf.loc[rf.iloc[:,0] == '01'].index[0]
noProdIndexRF = rf.loc[rf.iloc[:,0] == '03'].index[0]
servicesIndexRF = rf.loc[rf.iloc[:,0] == '04'].index[0]

prodCategoryRF = Category('Продовольственные товары')
noProdCategoryRF = Category('Непродовольственные товары')

for i in range(prodIndexRF+1, noProdIndexRF):
    cell = rf.iloc[i, 0]
    name = rf.iloc[i, 1]
    index = cell.split('.')
    
    if len(index) == 3:
        print('Group:', name)
        group = Group(name)
        prodCategoryRF.Append(group)
        lastList = group
    elif len(index) == 4:
        print('Subgroup', name)
        subgroup = Subgroup(name)
        group.Append(subgroup)
        lastList = subgroup
    elif len(index) == 6:
        print('Product:', name)
        product = Product(name)
        lastList.Append(product)

for i in range(noProdIndexRF+1, servicesIndexRF):
    cell = rf.iloc[i, 0]
    name = rf.iloc[i, 1]
    index = cell.split('.')
    
    if len(index) == 3:
        print('Group:', name)
        group = Group(name)
        noProdCategoryRF.Append(group)
        lastList = group
    elif len(index) == 4:
        print('Subgroup', name)
        subgroup = Subgroup(name)
        group.Append(subgroup)
        lastList = subgroup
    elif len(index) == 6:
        print('Product:', name)
        product = Product(name)
        lastList.Append(product)

rf = pd.read_excel('data.xls', header=2)

rf = rf.rename(columns={'Unnamed: 0': 'name'})
rf = clean(rf, 'name')

prodRF = prodCategoryRF.GetProducts()
noProdRF = noProdCategoryRF.GetProducts()

comparedProdRF = compareRF(prodRF, rf, rf.columns[4], rf.columns[1],
                        0.7, alpha=0.5, beta=0.25, gamma=0.25)

comparedNoProdRF = compareRF(noProdRF, rf, rf.columns[4], rf.columns[1],
                        0.7, alpha=0.6, beta=0.25, gamma=0.15)

prod_rb_rf = compare_rb_rf(prodRB, prodRF, 0.7, 0.4, 0.35, 0.25)

noProd_rb_rf = compare_rb_rf(noProdRB, noProdRF, 0.7, 0.2, 0.5, 0.25)

prod_rb_rf.to_excel('prod_rb_rf.xlsx')
noProd_rb_rf.to_excel('noProd_rb_rf.xlsx')
