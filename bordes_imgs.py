import cv2, numpy as np

banana = cv2.imread("C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/imgs/bananos.jpg")
gray = cv2.cvtColor(banana, cv2.COLOR_BGRA2GRAY)
#cv2.imshow('original', gray)

'''
Para detectar precisamente los bordes de la imagen se emplea el vector gradiente
Este detecta los cambios bruscos entre píxeles contiguos y marca solamente el borde
Tiene una componente horizontal gx y una vertica gy
Para detectar 'falsos bordes' se emplea el método de Sobel, que elimina el ruido
'''
#(imagen, tamaño de flotantes, [1,0] para gx / [0,1] para gy, tamaño del kernel (5x5))
gx = cv2.Sobel(gray, cv2.CV_64F, 1,0,5)
gy = cv2.Sobel(gray, cv2.CV_64F, 0,1,5)

#cv2.imshow('x', gx)
#cv2.imshow('y', gy)

#Traduce las coordenadas cartesianas a coordenadas polares
mag, _ = cv2.cartToPolar(gx, gy)

#Transforma la magnitud entre 0 y 255
mag = np.uint8(255 * mag / np.max(mag))

#Ya tenemos los bordes de la imagen
cv2.imshow('', mag)

cv2.waitKey(0)
cv2.destroyAllWindows()