Autores:
    Mateus Garicia Baiak    GRR20224378
    --------------- ADICIONE SEU NOME --------------------------

Arquivos/Diretórios:
    ImagensOriginais:
        Diretório com as 55 fotos antes do pré-processamento, ainda coloridas e na resolução original, porém já cortadas como quadrados
    ImagensPreProcessadas:
        Diretório com as 55 fotos depois de serem pré-processadas, em tons de cinza e redimencionadas para 512x512
    preProcessador.py:
        Script Python para transformar as fotos que estão em ImagensOriginais, pré-processalas e então salvar eças em ImagensPreProcessadas
    main.py:
        Programa Main que exeuta os filtros e realiza o agrupamento
    requirements.txt:
        Arquivo para auxiliar para a instalçao de dependências

Como executar:
    É recomendado o uso de um ambiente virtual
    Para criar o ambiente virtual:
        python3 -m venv .venv
    Depois entrar no ambiente:
        source .venv/bin/activate
    Depois instalar as dependências:
        pip install -r requirements.txt
    E por fim  para execuatar:
    - O Pre Processamento:
        python preProcessador.py
    - O Main:
        python main.py

Seleção das fotos:
    Foram selecionadas 55 fotos de deversos contextos: corredores, ruas, estádio, céu, chão, etc ... Escolhemos essas imagens tendo a expectativa de que as que estão mais próximas dos objetos ou que tem um padrão mais uniforme durante toda a imagem sejam as que sejam melhor classificadas, já aquelas que tem várias texturas pela imagem ou que são de ambientes mais abertos entender o quão bem esses métodos vão se sair.
