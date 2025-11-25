import cv2, numpy as np

banana = cv2.imread("C:/Users/damia/OneDrive/Escritorio/Curso Tensorflow/imgs/bananos.jpg")

cv2.imshow('original', banana)

#Genera una matriz de 3x3 toda de unos
kernel_3x3 = np.ones((3,3)) / (3*3)

#Le pasa a banana el filtro kernel 3x3 (reduce su definición, la difumina)
output_3x3 = cv2.filter2D(banana, -1, kernel_3x3)
cv2.imshow('filtro_3x3', output_3x3)

#A mayor sea el kernel mayor es el filtro
kernel_11x11 = np.ones((11,11)) / (11*11)
output_11x11 = cv2.filter2D(banana, -1, kernel_11x11)
cv2.imshow('filtro_11x11', output_11x11)

kernel_31x31 = np.ones((31,31)) / (31*31)
output_31x31 = cv2.filter2D(banana, -1, kernel_31x31)
cv2.imshow('filtro_31x31', output_31x31)

cv2.waitKey(0)
cv2.destroyAllWindows()