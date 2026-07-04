"""
Passo 1: Geração de Dados (Setup)
---------------------------------
Gera o arquivo dataset_plantas.csv com ~100 linhas simulando o histórico
de inspeções de plantas: características observadas pelos sensores e o
diagnóstico REAL (confirmado, por exemplo, por um agrônomo humano).

Este CSV é o "conhecimento histórico" que vai alimentar o treinamento do
Naive Bayes no Passo 2.
"""

import csv
import random
from sensores import ESTADOS, gerar_leitura_sensores

N_LINHAS = 100
SEED = 42
ARQUIVO_SAIDA = "dataset_plantas.csv"


def gerar_dataset(n_linhas=N_LINHAS, seed=SEED, caminho=ARQUIVO_SAIDA):
    random.seed(seed)

    linhas = []
    for _ in range(n_linhas):
        # distribuição das classes no histórico (não precisa ser 1/3 exato,
        # estufas reais costumam ter mais plantas saudáveis que doentes)
        estado_real = random.choices(ESTADOS, weights=[0.45, 0.30, 0.25])[0]
        leitura = gerar_leitura_sensores(estado_real)
        linhas.append({
            "cor_folha": leitura["cor_folha"],
            "umidade_solo": leitura["umidade_solo"],
            "tamanho": leitura["tamanho"],
            "diagnostico": estado_real,
        })

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        campos = ["cor_folha", "umidade_solo", "tamanho", "diagnostico"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas)

    print(f"[OK] Dataset gerado: {caminho} ({len(linhas)} linhas)")
    return linhas


if __name__ == "__main__":
    gerar_dataset()