import cv2
#Se obtiene el sombrero y se establecen sus caracteristicas 
sombrero=cv2.imread("sombrero1.jpg")
hs,ws,_=sombrero.shape
talla=2
proporciones=ws/hs

#se define el clasificador para detectar la cara
clasificador=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Se obtiene la cara y la convertimos a escala de grises para que sea detectada por
#el clasificador HaarCascades
img=cv2.imread("cara6.jpg")
img_byn=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
alto,ancho,_=img.shape

#se detecta la cara
caras=clasificador .detectMultiScale(img_byn)

#de acuerdo a la cara(s) obtenidas se detectan las medidas para definir la
#medida del sombrero y la posicion sobre la cual se colocara en la cabeza
for(x_cara,y_cara,anchoC,altoC) in caras:
    anchoS=int(anchoC*talla)
    altoS=int(anchoS/proporciones)
    sombrero=cv2.resize(sombrero,(anchoS,altoS))
    y_sombrero=y_cara-altoS
    x_sombrero=int(x_cara+anchoC/2-anchoS/2)
    #Se coloca el sombrero en la cabeza, obteniendo el area de interes de la cabeza
    #se coloca en el area de interes (roi) el sombrero a traves de la operacion logica
    #and, y al final se agrega de nueva cuenta esa porcion a la imagen original
    if x_sombrero>=0 and y_sombrero>=0 and x_sombrero+anchoS<=ancho and y_sombrero+altoS<=alto:
        roi=img[y_sombrero:y_sombrero+altoS,x_sombrero:x_sombrero+anchoS]
        roi=cv2.bitwise_and(sombrero, roi)
        img[y_sombrero:y_sombrero+altoS,x_sombrero:x_sombrero+anchoS]=roi
cv2.imshow("",img)
cv2.imshow("sombrero",sombrero)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
