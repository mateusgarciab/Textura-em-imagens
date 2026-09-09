Ele está uzando um ambiente virtual (chamado tarefa)

Não sei o quão chato é para instalar sozinho mas tente fazer o seguinte:

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python preProcessador.py