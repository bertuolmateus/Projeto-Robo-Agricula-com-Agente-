"""
Passo 3: Criação do Ambiente (Orientação a Objetos)
-----------------------------------------------------
Define:
  - Planta: uma célula do campo, com um estado REAL (oculto, "verdade
    absoluta" - como se soubéssemos o diagnóstico de um especialista) e uma
    leitura de sensores (o que o robô efetivamente enxerga, com ruído).
  - Estufa: uma matriz NxM de Plantas.

Guardar o estado REAL na Planta (além da leitura ruidosa) é o que permite,
no Passo 4/5, comparar a decisão do agente (baseada só na leitura dos
sensores) com a "verdade" e assim medir o quão bem o Naive Bayes se saiu
em campo - e não só no conjunto de teste do treinamento.
"""

import random
from sensores import ESTADOS, gerar_leitura_sensores


class Planta:
    def __init__(self, estado_real: str):
        self.estado_real = estado_real
        self.leitura_sensores = gerar_leitura_sensores(estado_real)
        self.foi_visitada = False
        self.acao_recebida = None
        self.diagnostico_dado_agente = None

    def __repr__(self):
        return f"Planta(real={self.estado_real}, sensores={self.leitura_sensores})"


class Estufa:
    """
    Representa o campo/estufa como uma matriz NxM. Cada célula contém uma
    Planta com estado gerado aleatoriamente (com uma distribuição de
    probabilidade configurável, simulando uma estufa real onde a maioria
    das plantas é saudável).
    """

    def __init__(self, linhas=5, colunas=5, seed=None,
                 pesos_estado=(0.5, 0.25, 0.25)):
        if seed is not None:
            random.seed(seed)

        self.linhas = linhas
        self.colunas = colunas
        self.grid = [
            [
                Planta(random.choices(ESTADOS, weights=list(pesos_estado))[0])
                for _ in range(colunas)
            ]
            for _ in range(linhas)
        ]

    def get_planta(self, i, j) -> Planta:
        return self.grid[i][j]

    def posicoes(self):
        """Gera as posições (i, j) em ordem de varredura tipo 'cobra' (boustrophedon),
        um padrão clássico de cobertura completa de área em robótica de campo."""
        for i in range(self.linhas):
            colunas_range = range(self.colunas) if i % 2 == 0 else range(self.colunas - 1, -1, -1)
            for j in colunas_range:
                yield (i, j)

    def resumo_estado_real(self):
        """Conta quantas plantas de cada estado real existem (útil para análise)."""
        contagem = {estado: 0 for estado in ESTADOS}
        for linha in self.grid:
            for planta in linha:
                contagem[planta.estado_real] += 1
        return contagem

    def imprimir_mapa(self, mostrar_estado_real=False):
        # Usamos letras simples (e não emojis) porque o terminal padrão do
        # Windows (cp1252) não consegue exibir caracteres unicode como 🟢🟡🔴,
        # o que gera UnicodeEncodeError.
        simbolos = {"Saudavel": "[S]", "Sede": "[X]", "Doente": "[D]"}
        for linha in self.grid:
            if mostrar_estado_real:
                print(" ".join(simbolos[p.estado_real] for p in linha))
            else:
                marca = []
                for p in linha:
                    if not p.foi_visitada:
                        marca.append("[ ]")
                    else:
                        marca.append(simbolos.get(p.diagnostico_dado_agente, "[?]"))
                print(" ".join(marca))