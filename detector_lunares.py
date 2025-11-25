import cv2, numpy as np

#PRE-PROCESAMIENTO Y SEGMENTACIÓN
#Preprocesamos la imagen en esta función y obtenemos las características
def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Con el threshold determinamos cuando es fondo y cuando es lunar
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    #Como el lunar es más oscuro que la piel nos interesan los valores que están por debajo del threshold
    #Si fuera más claro que la piel buscaríamos los valores que estuvieran por encima del threshold
    mask = np.uint8(1*(gray < threshold))

    #Así hacemos un promedio de la cantidad de cada componente que hay en la imagen con el mask
    #Conseguimos que sean números reales entre 0 y 1, es el porcentaje
    b = (1 / 255) * np.sum(img[:,:,0] * mask) / np.sum(mask)
    g = (1 / 255) * np.sum(img[:,:,1] * mask) / np.sum(mask)
    r = (1 / 255) * np.sum(img[:,:,2] * mask) / np.sum(mask)

    return [b, g, r]

#EXTRACCIÓN DE CARACTERÍSTICAS
#Dependencia para leer mejor archivos
import glob

#Ponemos las 2 rutas a las carpetas de entrenamiento
paths = ['C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/detector_lunares/datasetLunares/dysplasticNevi/train/',
         'C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/detector_lunares/datasetLunares/spitzNevus/train/']

labels = []     #Etiquetas
features = []   #Características

#Para cada path en paths
for label, path in enumerate(paths):
    #Para cada archivo en path que acabe en jpg
    for filename in glob.glob(path + "*.jpg"):
        #Cada imagen es un archivo
        img = cv2.imread(filename)
        #Mete en características las proporciones de colores de la imagen
        features.append(getFeatures(img))
        #Label es el id que se genera que relacionará imágenes con features
        #Las imágenes de la primera carpeta tendrán la etiqueta 0 y las de la segunda la 1
        labels.append(label)

#Los transformamos a arreglos de numpy para tratarlos mejor
features = np.array(features)
labels = np.array(labels)
#Labels normal quedaría 0 y 1 pero de esta manera quedaría así
#0 y 1 * 2 son 0 y 2 y -1 son -1 y 1
#Así tenemos -1 y 1
labels = 2 * labels - 1

#Visualizamos el dataset en el espacio de características
import matplotlib.pyplot as plt

#Creamos una figura de pyplot
fig = plt.figure()
#Añade una gráfica en 3D
ax = fig.add_subplot(111, projection = '3d')

#Para cada característica en el dataset de características
for i, feature_row in enumerate(features):
    #Si tiene la primera etiqueta
    if labels[i] == -1:
        #Se dibuja un * en el punto 3d de las cordenadas correspondientes a las 3 features que son RGB
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker='*', c='k') #k para negro
    else:
        ax.scatter(feature_row[0], feature_row[1], feature_row[2], marker='*', c='r') #r para rojo

#Seteamos los ejes
ax.set_xlabel('B')
ax.set_xlabel('G')
ax.set_xlabel('R')

#plt.show()  #Muestra 

#Podemos ver que es un conjuntos de datos no linealmente separable
#Calculamos el error en función de las constantes del hiperplano

#Cogemos solo 2 características para hacerlo en 2d
subFeatures = features[:,1::]   #1:: significa el de la posición 1 y los de después

#Instanciamos un array de perdidas
loss = []

#Para cada constante w1 de la primera feature
for w1 in np.linspace(-6, 6, 100):
    #Para cada constante w2 de la segunda feature
    for w2 in np.linspace(-6, 6, 100):
        totalError = 0
        #Para cada subFeature
        for i, feature_row in enumerate(subFeatures):
            #Fórmula para sacar el error de la muestra
            #(w1 * x1 + w2 * x2 + w... * x... - yi)**2
            sample_error = (w1 * feature_row[0] + w2 * feature_row[1] - labels[i])**2
            #Se suma el valor de la muestra al error total
            totalError += sample_error
        #Se suma al array
        loss.append([w1, w2, totalError])

#Lo transformamos a arreglo de numpy para tratarlo mejor
loss = np.array(loss)

#Importamos para graficar mapas
import matplotlib.cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

#Generamos el plano dados 3 puntos
ax1.plot_trisurf(loss[:,0],loss[:,1],loss[:,2], cmap = cm.jet, linewidth=0)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('loss')

#plt.show()     #Muestra

#Encontramos que el error tiene que ser el punto más bajo del gráfico
#Con esto buscamos el hiperplano w que separa las 2 clases de forma óptima
#A es una sumatoria y la iniciamos en zeros porque va a ser una matriz acumuladora
A = np.zeros((4,4)) #4x4

#b también es una sumatoria y también es una matriz acumuladora
b = np.zeros((4,1)) #4x1

#Para cada feature
for i, feature_row in enumerate(features):
    #Añadimos un 1 al inicio
    x = np.append([1], feature_row)
    x = x.reshape((4,1)) #4x1
    y = labels[i]
    # A = sum(x * x.traspuesta)
    A = A + x * x.T #Se añade a A la matriz por su matriz traspuesta
    # b = sum(x * y)
    b = b + x * y

#Calculamos la inversa de A que es A**-1
invA = np.linalg.inv(A)

#w = A**-1 * b
W = np.dot(invA, b) #dot multiplica matrices

#Graficamos X y Y
X = np.arange(0,1,0.1)
Y = np.arange(0,1,0.1)

X, Y =np.meshgrid(X, Y)

#W[3] * Z + W[1] * X + W[2] * Y + W[0] = 0
#Despejamos z y hayamos
Z = -(W[1] * X + W [2] * Y + W[0]) / W[3]

ax.plot_surface(X, Y, Z, cmap = cm.Blues)

#plt.show() #Muestra

#Muestra con errores
#Error de entrenamiento
#Predecimos si será correcto o no y devolverá 1 o 0
prediction = 1 * (W[0] + np.dot(features, W[1::])) > 0

#Lo modificamos para que de 1 y -1
prediction = 2 * prediction - 1

#Haya el error
error = np.sum(prediction != labels.reshape(-1,1))/len(labels)
error_percentage = error * 100

#Haya la efectividad
accuracy = 1 - error
accuracy_percentage = accuracy * 100

#Predicción para una imagen
def resultador(path_img):
    img = cv2.imread(path_img)
    feature_vector = np.array(getFeatures(img))
    result = np.sign(W[0]+np.dot(feature_vector, W[1::]))
    return result

#Mostramos el resultado
path_base_img = 'C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/detector_lunares/datasetLunares/spitzNevus/train/'
path_concrete = 'spitzNevus6.jpg'
path_img = path_base_img + path_concrete

if resultador(path_img) == -1 :
    print("Es un displasticNevi")
else :
    print("Es un spitzNevus")

print(f"El error puede ser del {error_percentage}%")
print(f"La precisión puede ser del {accuracy_percentage}%")