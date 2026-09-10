import os
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# BANCO DE FILTROS (ORIENTACOES, CIRCULARES E BORDAS)
def obter_banco_filtros():
    """
    Constroi 8 filtros de textura:
        4 Gabors: 0 (Horizontal), 45, 90 (Vertical), 135 graus.
        2 Circulares: Laplacian of Gaussian (LoG) e Difference of Gaussians (DoG).
        Bordas: Sobel Dx e Sobel Dy.
    """
    filtros = []
    k_size = 21
    sigma, lambd, gamma, psi = 3.0, 7.0, 0.5, 0

    # Filtros de 1 a 4. Filtros de Gabor
    orientacoes = [("Gabor 0 Graus (H)", 0),
                   ("Gabor 45 Graus", np.pi / 4),
                   ("Gabor 90 Graus (V)", np.pi / 2),
                   ("Gabor 135 Graus", 3 * np.pi / 4)]
    
    for nome, theta in orientacoes:
        gabor = cv2.getGaborKernel((k_size, k_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filtros.append((nome, gabor))

    # Filtro 5 Circular: Laplacian of Gaussian (LoG)
    x = np.linspace(-3, 3, k_size)
    y = np.linspace(-3, 3, k_size)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2
    log_k = (1 - r2 / 2) * np.exp(-r2 / 2)
    log_k = (log_k - log_k.mean()).astype(np.float32)
    filtros.append(("Circular LoG", log_k))

    # 6. Circular: Difference of Gaussians (DoG)
    g1 = cv2.getGaussianKernel(k_size, 1.0)
    g2 = cv2.getGaussianKernel(k_size, 3.0)
    dog_k = (np.outer(g1, g1) - np.outer(g2, g2)).astype(np.float32)
    dog_k = (dog_k - dog_k.mean()).astype(np.float32)
    filtros.append(("Circular DoG", dog_k))

    # Filtros 7 e 8. Bordas Direcionais (Sobel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    filtros.append(("Sobel Dx", sobel_x))
    filtros.append(("Sobel Dy", sobel_y))

    return filtros

# EXTRACAO MULTIESCALA DE CARACTERISTICAS (24 DIMENSOES)
def extrair_caracteristicas_24d(img_cinza, banco_filtros, tamanho_janela=31):
    #Aplica 8 filtros em 3 escalas de imagem (Piramide Gaussiana: 512x512, 256x256, 128x128).
    h, w = img_cinza.shape
    
    # 3 Escalas via Piramide Gaussiana
    escala1 = img_cinza.astype(np.float32)
    escala2 = cv2.pyrDown(escala1)
    escala3 = cv2.pyrDown(escala2)
    escalas = [escala1, escala2, escala3]

    respostas = []

    for img_esc in escalas:
        for _, kernel in banco_filtros:
            # Filtragem
            filtrada = cv2.filter2D(img_esc, cv2.CV_32F, kernel)
            
            # Energia da textura
            energia = np.abs(filtrada)
            
            # Suavizacao por janela deslizante continua
            media_local = cv2.blur(energia, (tamanho_janela, tamanho_janela))
            
            # Redimensiona para alinhamento espacial de 512x512
            if media_local.shape != (h, w):
                media_local = cv2.resize(media_local, (w, h), interpolation=cv2.INTER_LINEAR)
                
            respostas.append(media_local)

    mapa_24d = np.stack(respostas, axis=-1)  # (512, 512, 24)
    vetor_global_24d = mapa_24d.mean(axis=(0, 1))  # (24,)
    
    return mapa_24d, vetor_global_24d

# NORMALIZACAO E AGRUPAMENTO (K-MEANS)
def normalizar_zscore(X):
    """ Padroniza os dados (Z-Score) sem bibliotecas externas: (X - u) / std """
    media = np.mean(X, axis=0, keepdims=True)
    desvio = np.std(X, axis=0, keepdims=True)
    desvio[desvio == 0] = 1.0
    return (X - media) / desvio

def agrupar_kmeans(dados_24d, n_grupos=4):
    """ Agrupa vetores 24D usando o cv2.kmeans nativo com distancia euclidiana """
    X_norm = normalizar_zscore(dados_24d).astype(np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, rotulos, _ = cv2.kmeans(
        data=X_norm,
        K=n_grupos,
        bestLabels=None,
        criteria=criterio,
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )
    return rotulos.flatten()

# PLOTS VISUAIS E GERACAO DE RESULTADOS
def salvar_visualizacao_banco(banco_filtros, pasta_saida):
    """ Salva imagem do banco de filtros """
    plt.figure(figsize=(14, 6))
    for i, (nome, k) in enumerate(banco_filtros):
        plt.subplot(2, 4, i + 1)
        plt.imshow(k, cmap='jet')
        plt.title(nome, fontsize=10)
        plt.axis('off')
    plt.suptitle("Banco de Filtros de Textura (8 Kernels)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, "banco_de_filtros.png"), dpi=150)
    plt.close()

def salvar_segmentacao_individual(img_cinza, mapa_24d, vetor_24d, nome_arquivo, pasta_saida, n_grupos=4):
    """ Gera o painel de 3 partes: Imagem, Segmentacao Colorida e Grafico do Vetor 24D """
    h, w, d = mapa_24d.shape
    X = mapa_24d.reshape(-1, d)
    
    # Agrupamento local das regioes
    rotulos = agrupar_kmeans(X, n_grupos=n_grupos)
    mascara = rotulos.reshape(h, w)

# Paleta de cores RGB para os 7 grupos da segmentacao local
    paleta = np.array([
        [255, 99, 71],    # Grupo 0: Vermelho
        [50, 205, 50],    # Grupo 1: Verde Lime
        [30, 144, 255],   # Grupo 2: Azul
        [255, 215, 0],    # Grupo 3: Amarelo
        [147, 112, 219],  # Grupo 4: Roxo
        [0, 206, 209],    # Grupo 5: Ciano
        [255, 140, 0]     # Grupo 6: Laranja
    ], dtype=np.uint8)

    seg_colorida = paleta[mascara % len(paleta)]

    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title(f"Original: {nome_arquivo}", fontsize=10)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(seg_colorida)
    plt.title(f"Segmentacao por Textura ({n_grupos} Grupos)", fontsize=10)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.bar(range(1, 25), vetor_24d, color='teal')
    plt.xlabel("Dimensoes (1 a 24)")
    plt.ylabel("Resposta de Energia")
    plt.title("Assinatura de Textura 24D", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f"seg_{nome_arquivo}"), dpi=150)
    plt.close()

def salvar_agrupamento_global(imagens_cinza, nomes, rotulos_globais, pasta_saida):
    """ 
    Gera a grade final com molduras coloridas ordenando DINAMICAMENTE 
    todas as imagens encontradas (ex: 55 imagens).
    """
    total = len(imagens_cinza)
    colunas = 8
    linhas = math.ceil(total / colunas)

    # Ajusta a altura da figura de acordo com a quantidade de linhas
    fig, axes = plt.subplots(linhas, colunas, figsize=(18, 2.5 * linhas))
    
    # Se houver apenas 1 linha ou 1 imagem, garante que axes seja um array plano
    if total > 1:
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    cores_hex = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

    for i in range(len(axes)):
        if i < total:
            grupo = rotulos_globais[i]
            axes[i].imshow(imagens_cinza[i], cmap='gray')
            axes[i].set_title(f"{nomes[i]}\nGrupo {grupo + 1}", fontsize=8, pad=2)
            
            # Adiciona a moldura colorida
            for spine in axes[i].spines.values():
                spine.set_edgecolor(cores_hex[grupo % len(cores_hex)])
                spine.set_linewidth(3)
                
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        else:
            # Oculta quadros sobressalentes caso o total nao preencha a grade perfeitamente
            axes[i].axis('off')

    plt.suptitle(f"Agrupamento Global do Dataset (Distancia Euclidiana 24D em {total} Imagens)", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f"agrupamento_global_{total}_imagens.png"), dpi=200)
    plt.close()

# EXECUCAO PRINCIPAL (MAIN PIPELINE)
if __name__ == "__main__":
    N_GRUPOS = 7
    pasta_entrada = "ImagensPreProcessadas"
    pasta_saida = "Resultados"
    os.makedirs(pasta_entrada, exist_ok=True)
    os.makedirs(os.path.join(pasta_saida, "Filtros"), exist_ok=True)
    os.makedirs(os.path.join(pasta_saida, "Segmentacoes"), exist_ok=True)
    os.makedirs(os.path.join(pasta_saida, "AgrupamentoGlobal"), exist_ok=True)

    print("=== PIPELINE DE PROCESSAMENTO DE TEXTURA ===")
    
    # Banco de Filtros
    banco_filtros = obter_banco_filtros()
    salvar_visualizacao_banco(banco_filtros, os.path.join(pasta_saida, "Filtros"))
    print("[1/4] Banco de Filtros gerado.")

    # Leitura dos Arquivos (Busca extensoes comuns: .jpg, .jpeg, .png)
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    arquivos = []
    for ext in extensoes:
        arquivos.extend(glob.glob(os.path.join(pasta_entrada, ext)))
    arquivos = sorted(list(set(arquivos)))

    print(f"[2/4] Encontradas {len(arquivos)} imagens em '{pasta_entrada}'.")

    if len(arquivos) == 0:
        print(f"ERRO: Nenhuma imagem encontrada em '{pasta_entrada}'. Verifique os arquivos e tente novamente.")
        exit()

    vetores_globais = []
    imagens_carregadas = []
    nomes_arquivos = []

    # Processamento Individual
    print("[3/4] Extraindo Vetores 24D e Segmentando...")
    for idx, caminho in enumerate(arquivos):
        nome_arq = os.path.basename(caminho)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"      Aviso: Nao foi possivel ler {nome_arq}. Pulando...")
            continue

        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512))

        mapa_24d, vetor_24d = extrair_caracteristicas_24d(img, banco_filtros)
        
        salvar_segmentacao_individual(
            img, mapa_24d, vetor_24d, nome_arq, 
            os.path.join(pasta_saida, "Segmentacoes"), n_grupos=N_GRUPOS
        )

        vetores_globais.append(vetor_24d)
        imagens_carregadas.append(img)
        nomes_arquivos.append(nome_arq)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(arquivos):
            print(f"      Processadas {idx + 1}/{len(arquivos)} imagens.")

    # Agrupamento Global do Dataset
    print("[4/4] Realizando Agrupamento Global do Dataset...")
    X_global = np.array(vetores_globais)
    rotulos_globais = agrupar_kmeans(X_global, n_grupos=N_GRUPOS)

    salvar_agrupamento_global(
        imagens_carregadas, nomes_arquivos, rotulos_globais, 
        os.path.join(pasta_saida, "AgrupamentoGlobal")
    )

    print("=== CONCLUIDO COM SUCESSO! Verifique a pasta 'Resultados/' ===")