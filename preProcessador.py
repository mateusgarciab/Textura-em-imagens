import cv2

for i in range(1, 56):
    caminho = f"ImagensOriginais/{i}.jpg"
    print(caminho)

    imagem = cv2.imread(caminho)

    imagem = cv2.resize(imagem, (512, 512))

    """ Mostra a imagem redimencionada """
    """ cv2.imshow("Imagem", imagem)
    cv2.waitKey(0) """

    vCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    """ Mostra a imagem em tons de cinza """
    """ cv2.imshow("Cinza", vCinza)
    cv2.waitKey(0) """

    caminhoEscrita = f"ImagensPreProcessadas/{i}.jpg"
    cv2.imwrite(caminhoEscrita,vCinza)