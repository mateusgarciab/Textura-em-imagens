import cv2
import numpy as np

vetor = []

for i in range(1, 56):
    caminho = f"ImagensPreProcessadas/{i}.jpg"
    imagemEscala1 = cv2.imread(caminho)
    imagemEscala2 = cv2.resize(imagemEscala1, (256, 256))
    imagemEscala3 = cv2.resize(imagemEscala1, (128, 128))

    vetorRodada = []

    """ cv2.imshow("P1", imagemEscala1)
    cv2.waitKey(0)

    cv2.imshow("P2", imagemEscala2)
    cv2.waitKey(0)

    cv2.imshow("P3", imagemEscala3)
    cv2.waitKey(0) """

    """ ele passou 5 filtros para fazer, em 3 escalas, 3 * 5 = 15, o vetor precisa ter tamanho 24, não sei como vamos completar isso """
    kernel_horizontal = np.array([[-1, -1, -1], 
                                  [ 2,  2,  2], 
                                  [-1, -1, -1]], dtype=np.float32) 

    resposta = cv2.filter2D(imagemEscala1, cv2.CV_32F, kernel_horizontal)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala2, cv2.CV_32F, kernel_horizontal)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala3, cv2.CV_32F, kernel_horizontal)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)


    """ Com ctz dava para fazer um for que seria mais facil do q isso, amanhã eu vejo """
    kernel_vertical = np.array([[-1, 2, -1], 
                                [-1, 2, -1], 
                                [-1, 2, -1]], dtype=np.float32) 
    resposta = cv2.filter2D(imagemEscala1, cv2.CV_32F, kernel_vertical)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala2, cv2.CV_32F, kernel_vertical)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala3, cv2.CV_32F, kernel_vertical)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)



    kernel_45 = np.array([[ 2, -1, -1],
                          [-1,  2, -1],
                          [-1, -1,  2]], dtype=np.float32)
    resposta = cv2.filter2D(imagemEscala1, cv2.CV_32F, kernel_45)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala2, cv2.CV_32F, kernel_45)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala3, cv2.CV_32F, kernel_45)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)



    kernel_135 = np.array([[-1, -1,  2],
                           [-1,  2, -1],
                           [ 2, -1, -1]], dtype=np.float32)
    resposta = cv2.filter2D(imagemEscala1, cv2.CV_32F, kernel_135)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala2, cv2.CV_32F, kernel_135)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    resposta = cv2.filter2D(imagemEscala3, cv2.CV_32F, kernel_135)
    caracteristica = np.mean(np.abs(resposta))
    vetorRodada.append(caracteristica)

    print(vetorRodada)
    print(len(vetorRodada))
    vetor.append(vetorRodada)

print(vetor)