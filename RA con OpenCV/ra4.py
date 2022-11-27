import cv2
sombrero=cv2.imread("sombrero5.jpg")
altoS,anchoS,_=sombrero.shape
#talla=2
talla=1.4
proporciones=anchoS/altoS

#area del contorno de la cara, se puede probar con varios valores y dejar el que
#mejor se adecue
areaMin=10000
#se define el clasificador para detectar la cara
clasificador=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camara=cv2.VideoCapture(0)
if not camara.isOpened():
    print("No es posible abrir la camara")
    exit()
anchoVentana=int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
altoVentana=int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame=camara.read()
    if not ret:
        print("no es posible obtener la imagen")
        break
    frame_byn=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #se genera la deteccion de la cara
    caras=clasificador .detectMultiScale(frame_byn)
    for(x_cara,y_cara,anchoC,altoC) in caras:
        #se definen las medidas y posiciones para colorar el sombrero en la cabeza
        anchoS=int(anchoC*talla)
        altoS=int(anchoS/proporciones)
        sombrero=cv2.resize(sombrero,(anchoS,altoS))
        y_sombrero=y_cara-altoS
        x_sombrero=int(x_cara+anchoC/2-anchoS/2)
        #Se coloca el sombrero en la cabeza, obteniendo el area de interes de la cabeza
        #se coloca en el area de interes (roi) el sombrero a traves de la operacion logica
        #and, y al final se agrega de nueva cuenta esa porcion a la imagen original
        if x_sombrero>=0 and y_sombrero>=0 and x_sombrero+anchoS<=anchoVentana and y_sombrero+altoS<=altoVentana:
            roi=frame[y_sombrero:y_sombrero+altoS,x_sombrero:x_sombrero+anchoS]
            roi=cv2.bitwise_and(sombrero, roi)
            frame[y_sombrero:y_sombrero+altoS,x_sombrero:x_sombrero+anchoS]=roi
    cv2.imshow("",frame)
    #cv2.imshow("sombrero",sombrero)
    if cv2.waitKey(10)==ord('q'):
        break
camara.release()
cv2.destroyAllWindows()

    
