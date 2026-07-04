"""
sensores.py
-----------
Módulo central que simula a "física" do problema: dado o estado REAL de uma
planta (Saudavel, Sede, Doente), gera leituras de sensores RUIDOSAS
(cor da folha, umidade do solo, tamanho).

Esse mesmo gerador é usado em dois lugares:
1. gerar_dataset.py  -> cria o histórico (CSV) para treinar o Naive Bayes.
2. estufa.py         -> cria as plantas "reais" que o robô vai encontrar em campo.

Mantendo a lógica num único lugar garantimos que o modelo é treinado com
o mesmo tipo de distribuição de dados que ele vai encontrar em produção
(uma boa prática de ML: dados de treino e de uso devem vir da mesma distribuição).
"""

import random

ESTADOS = ["Saudavel", "Sede", "Doente"]
CORES = ["Verde", "Amarela", "Marrom"]
TAMANHOS = ["Pequeno", "Medio", "Grande"]


def gerar_leitura_sensores(estado_real: str) -> dict:
    """
    Gera uma leitura de sensores (com ruído) condicionada ao estado real da planta.

    Regras (o "mundo real" que o Naive Bayes terá que aprender a partir dos dados):

    - Saudavel: folha tende a Verde, umidade média/alta, tamanho Medio/Grande.
    - Sede:     folha tende a Verde ou Amarela (começa a amarelar), umidade BAIXA.
    - Doente:   folha tende a Amarela ou Marrom, umidade pode ser normal ou alta
                (excesso de água favorece fungo/doença), tamanho tende a Pequeno.

    Ruído: em ~15% dos casos, a cor "escapa" do padrão esperado, simulando erro
    de sensor / iluminação ruim / variação natural. Isso é o que faz o problema
    ser probabilístico e não determinístico -> justifica o uso de Naive Bayes.
    """
    ruido = 0.15

    if estado_real == "Saudavel":
        cor = random.choices(CORES, weights=[0.80, 0.15, 0.05])[0]
        umidade = round(random.gauss(65, 10), 1)
        tamanho = random.choices(TAMANHOS, weights=[0.15, 0.45, 0.40])[0]

    elif estado_real == "Sede":
        cor = random.choices(CORES, weights=[0.45, 0.45, 0.10])[0]
        umidade = round(random.gauss(20, 8), 1)
        tamanho = random.choices(TAMANHOS, weights=[0.35, 0.45, 0.20])[0]

    elif estado_real == "Doente":
        cor = random.choices(CORES, weights=[0.10, 0.45, 0.45])[0]
        umidade = round(random.gauss(60, 20), 1)  # pode ser normal ou alta (fungo)
        tamanho = random.choices(TAMANHOS, weights=[0.55, 0.30, 0.15])[0]
    else:
        raise ValueError(f"Estado desconhecido: {estado_real}")

    # aplica ruído adicional puro (erro de sensor independente do estado)
    if random.random() < ruido:
        cor = random.choice(CORES)

    umidade = max(0, min(100, umidade))  # limita entre 0 e 100%

    return {
        "cor_folha": cor,
        "umidade_solo": umidade,
        "tamanho": tamanho,
    }