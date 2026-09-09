import cv2
import numpy as np

vetor = []

""" Horizontal, Vertical, 45º, 135º, Circular, Alta frequência, Circular Maior """
vetKernels = [ [[-1, -1, -1], 
                [ 2,  2,  2], 
                [-1, -1, -1]],

                [[-1, 2, -1], 
                 [-1, 2, -1], 
                 [-1, 2, -1]],

                [[ 2, -1, -1],
                 [-1,  2, -1],
                 [-1, -1,  2]],

                [[-1, -1,  2],
                 [-1,  2, -1],
                 [ 2, -1, -1]],

                [[-1,  -1,  -1,  -1, -1],
                 [-1,   1,   1,   1,  -1],
                 [-1,   1,   4,   1,  -1],
                 [-1,   1,   1,   1,  -1],
                 [-1,  -1,  -1,  -1,  -1]],

                [[ 0, -1,  0],
                 [-1,  4, -1],
                 [ 0, -1,  0]],

                [[-1, -1, -1, -1, -1, -1, -1],
                 [-1, -1,  0,  0,  0, -1, -1],
                 [-1,  0,  1,  1,  1,  0, -1],
                 [-1,  0,  1,  8,  1,  0, -1],
                 [-1,  0,  1,  1,  1,  0, -1],
                 [-1, -1,  0,  0,  0, -1, -1],
                 [-1, -1, -1, -1, -1, -1, -1]]
                               ]

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

    for kernel in vetKernels:
        kernelRodada = np.array(kernel, dtype=np.float32)

        resposta = cv2.filter2D(imagemEscala1, cv2.CV_32F, kernelRodada)
        caracteristica = np.mean(np.abs(resposta))
        vetorRodada.append(caracteristica)
    
        resposta = cv2.filter2D(imagemEscala2, cv2.CV_32F, kernelRodada)
        caracteristica = np.mean(np.abs(resposta))
        vetorRodada.append(caracteristica)
    
        resposta = cv2.filter2D(imagemEscala3, cv2.CV_32F, kernelRodada)
        caracteristica = np.mean(np.abs(resposta))
        vetorRodada.append(caracteristica)



    
    """ é bom normalizar o vetor """
    print(vetorRodada)
    print(len(vetorRodada))
    vetor.append(vetorRodada)

""" print(vetor) """