# Segundo avance proyecto del curso "Tecnicas de inteligencia artificial"
# Grupo 1: Aislamiento de caracteres con "Asta ascendente"
# Andres Mauricio Rodriguez Garzón
# Andrés Sebastián Céspedes Cubides

# Análisis de las imágenes obtenidas
# Preproceso de imágenes
# Separación de líneas y párrafos
# Corrección de inclinación
# Filtros para ruido

import numpy as np
from matplotlib import pyplot as plt
import cv2


imgC = cv2.imread('pagina14.tif', 1)  # Carga imagen en color

numcols = imgC.shape[1]
numrows = imgC.shape[0]
pltScale = 3
# fig= plt.figure(figsize=(pltScale*10, pltScale*5))
# imgplot = plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB), plt.title('Página 14')
# plt.show()

# #########Umbralización usando el método de Otsu:########

gray = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
gaussianBlur = cv2.GaussianBlur(gray, (3, 3), -1)  # Blur gaussiano pre-otsu
_, otsu = cv2.threshold(gaussianBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)  # Adaptative threshold
imgToSegmentate = th3 + otsu

# fig = plt.figure(figsize=(pltScale*10, pltScale*5))
# imgplot = plt.imshow(imgToSegmentate, cmap='gray'), plt.title('Adaptativo + Otsu')
# plt.show()

# ######################Segmentación#######################

# 1. Colapso horizontal de la imágen:

whiteLevels = np.zeros((1, imgToSegmentate.shape[0]))

whiteLevels = np.sum(imgToSegmentate, axis=1)
whiteLevels.astype(int)

# Separación de líneas:
# Calcular el cambio de valor de negro en cada linea de pixels

linesSeparation = []

loopLength = len(whiteLevels) - 1

for i in range(loopLength):
    difference = int(whiteLevels[i]) - int(whiteLevels[i + 1])

    if difference > 11000:
        linesSeparation.append(i)
    difference = 0


lined = np.zeros(imgToSegmentate.shape)

# for i in range(len(linesSeparation)):
#     horIndex = linesSeparation[i]
#     lined = cv2.line(imgToSegmentate, (0, horIndex), (1550, horIndex), 100, thickness=1, lineType=8, shift=0)

# Una vez separadas las líneas buscamos los caracteres que sobresalgan de esta línea:
# Líneas superiores:

size = True
i = 2

while size:
    if i+2 == len(linesSeparation):
        size = False
    difference = abs(int(linesSeparation[i-1]) - int(linesSeparation[i-2]))

    if difference < 20:
        difference = abs(int(linesSeparation[i]) - int(linesSeparation[i - 2]))
        linesSeparation.pop(i-1)
        if difference > 10:
            i = i + 1
    else:
        i = i+1






# Luego de obtener las líneas superiores se busca identificar cada caracter con asta ascendente
# Se mide una distancia encima de las líneas, horizontalmente y si hay una diferencia muy grande en las diferencias
# Se toma como un caracter con asta ascendente


umbral = 10
vertDistance = 5
indexes = np.zeros((len(linesSeparation), ))

# Medición de diferencias en líneas horizontales
# Coordenadas de caracteres con asta ascendente

charactersList = []
midList = []

puntos = []
jAnt = 0

for i in range(len(linesSeparation)):

    for j in range(imgToSegmentate.shape[1]-1):
        currentPix = imgToSegmentate[linesSeparation[i]-vertDistance, j]
        nextPix = imgToSegmentate[linesSeparation[i]-vertDistance, j+1]
        difference = abs(int(nextPix) - int(currentPix))

        if difference > 150:

            midList.append(j)
            if abs(j - jAnt) > 15:
                puntos.append((j, linesSeparation[i]))
                jAnt = j
            else:
                jAnt = jAnt

    charactersList.append((linesSeparation[i], list(midList)))
    midList.clear()

# Limpieza de las líneas horizontales

for j in range(len(linesSeparation)):
    size = True
    i = 2
    while size:
        if i+2 == len(charactersList[j][1]):
            size = False
        difference = abs(int(charactersList[j][1][i-1]) - int(charactersList[j][1][i-2]))

        if difference < 3:
            # print(i)
            difference = abs(int(charactersList[j][1][i]) - int(charactersList[j][1][i - 2]))
            charactersList[j][1].pop(i-1)
            if difference > 2:
                i = i + 1
        else:
            i = i+1


for i in range(len(linesSeparation)):
    horIndex = linesSeparation[i]
    lined = cv2.line(imgToSegmentate, (0, horIndex), (1550, horIndex), 100, thickness=1, lineType=8, shift=0)
    for j in range(len(charactersList[i][1])):
        verIndex = charactersList[i][1][j]
        lined = cv2.line(imgToSegmentate, (verIndex-1, horIndex), (verIndex-1, horIndex-20), 100, thickness=1,
                         lineType=8, shift=0)



# fig = plt.figure(figsize=(pltScale*10, pltScale*5))
# imgplot = plt.imshow(lined, cmap='gray'), plt.title('lineas de párrafo')
# plt.show()
# Cercamiento de los caracteres
# Se toma un ancho de caracter predeterminado de 18 pixels
print(len(puntos))

vis = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB).copy()

for p in range(2):
    x = puntos[p][0]
    y = puntos[p][1]
    cv2.rectangle(vis, (x - 20, y - 30), (x + 20, y + 30), (0, 255, 0), 2)
    roi = imgC[y - 30:y + 30, x - 20:x + 20]
    cv2.imwrite('crop/' + str(p) + '.tif', roi)

testChar = cv2.imread('crop/1.tif', 1)  # Carga imagen en color
# testChar = testChar[5:-5, 5:-5]
mser = cv2.MSER_create(_delta=6, _min_area=100, _max_area=500, _max_variation=0.1)
regions, bboxes = mser.detectRegions(testChar)
hulls = [cv2.convexHull(p) for p in regions]
cv2.polylines(testChar, hulls, 1, (0, 255, 0))

fig = plt.figure(figsize=(pltScale*10, pltScale*5))
imgplot = plt.imshow(cv2.cvtColor(testChar, cv2.COLOR_BGR2RGB)), plt.title('lineas de párrafo')
plt.show()

print(len(regions))
