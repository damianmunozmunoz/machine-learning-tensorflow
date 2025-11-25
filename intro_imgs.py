#Importamos el lector de imágenes para python, numpy y matplotlib
import cv2, numpy as np, matplotlib.pyplot as plt

#Recibe el path de la imagen
banana = cv2.imread("C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/imgs/bananos.jpg")

#Obtenemos los componentes de colores de la matriz que nos devuelve openCV
b = banana[:, :, 0] #Componente azul
g = banana[:, :, 1] #Componente verde
r = banana[:, :, 2] #Componente rojo

#IMAGEN A COLOR
#Mostramos la imagen obtenida
#cv2.imshow('', banana) #El primer argumento es para darle un nombre con el que referirse a ella
#cv2.waitKey(0)
#cv2.destroyAllWindows() #Destruye las ventanas

#IMAGEN EN ESCALA DE GRISES
#Muestra la imagen en escala de grises
#(imagen inicial, transformación)
img_gray = cv2.cvtColor(banana, cv2.COLOR_BGR2GRAY) #Muestra la imagen en escala de grises
#cv2.imshow('', img_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#IMAGEN BINARIA
#uint8 lo transforma en un entero de 8 bits (max 255)
#Busca en la imagen de escala de grises que bits tienen un valor mayor o menor a x
x = 220
img_binary = np.uint8(255 * (img_gray < x))
#cv2.imshow('', img_binary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Multiplica cada valor de la posición de img_gray con su respectivo en img_binary
gray_segmentada = np.uint8(img_gray * (img_binary / 255))

#cv2.imshow('', gray_segmentada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

seg_color = banana.copy()
#Segmenta las variables de colores
seg_color[:, :, 0] = np.uint8(b * (img_binary / 255))
seg_color[:, :, 1] = np.uint8(g * (img_binary / 255))
seg_color[:, :, 2] = np.uint8(r * (img_binary / 255))

#cv2.imshow('', seg_color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Histograma que indica la frecuencia que hay de píxeles de cada color
#.flatten() convierte los píxeles de una imagen en un array
#bins especifica el número de columnas que va a tener el histograma
plt.hist(img_gray.flatten(), bins=15)
#plt.show()

#Con el método de otsu conseguimos sacar los bits que difieren entre el fondo y lo importante
threshold_otsu, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binary_otsu = np.uint8(255 * (img_gray < threshold_otsu))

#cv2.imshow('', binary_otsu)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

lunar = cv2.imread("C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/imgs/lunar.jpg")
lunar_gray = cv2.cvtColor(lunar, cv2.COLOR_BGR2GRAY)

threshold_lunar_otsu, _ = cv2.threshold(lunar_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

lunar_otsu = np.uint8(255 * (lunar_gray < threshold_lunar_otsu))

#cv2.imshow('', lunar_otsu)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

