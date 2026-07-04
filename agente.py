"""
Passo 4: Criação do Agente Híbrido
------------------------------------
AgenteAgricultor: um agente reativo baseado em utilidade com o clássico
loop de IA:

        PERCEPÇÃO -> RACIOCÍNIO (Naive Bayes) -> AÇÃO

Percepção:  ler os sensores da planta na célula atual (cor, umidade, tamanho).
Raciocínio: passar a leitura para o ModeloPlantas (Naive Bayes), que devolve
            o diagnóstico mais provável e as probabilidades de cada classe.
Ação:       escolher a ação correspondente ao diagnóstico, mas SOMENTE se o
            agente tiver recursos suficientes (água/bateria) - caso contrário,
            ele toma a decisão racional de adiar/pular a ação e registrar o
            problema, em vez de "fingir" que agiu.

Note que este arquivo NÃO importa nada do scikit-learn diretamente - ele só
conhece a interface `modelo.prever(leitura) -> (diagnostico, probabilidades)`.
Essa é a separação de responsabilidades exigida no critério de arquitetura:
lógica de ML fica inteiramente em treinar_modelo.py.
"""

from estufa import Estufa


class AgenteAgricultor:
    ACAO_POR_DIAGNOSTICO = {
        "Saudavel": "Ignorar",
        "Sede": "Regar",
        "Doente": "Aplicar Defensivo",
    }

    CUSTO_BATERIA_POR_MOVIMENTO = 1
    CUSTO_AGUA_POR_REGA = 10
    CUSTO_DEFENSIVO_POR_APLICACAO = 5
    CUSTO_BATERIA_POR_ACAO = 2

    def __init__(self, modelo, agua_max=100, bateria_max=100, defensivo_max=50):
        self.modelo = modelo
        self.agua = agua_max
        self.bateria = bateria_max
        self.defensivo = defensivo_max

        self.log = []
        self.acertos = 0
        self.erros = 0
        self.acoes_puladas_por_falta_recurso = 0

    # ---------- PERCEPÇÃO ----------
    def perceber(self, planta):
        return planta.leitura_sensores

    # ---------- RACIOCÍNIO ----------
    def raciocinar(self, leitura_sensores):
        diagnostico, probabilidades = self.modelo.prever(leitura_sensores)
        return diagnostico, probabilidades

    # ---------- AÇÃO ----------
    def decidir_acao(self, diagnostico):
        return self.ACAO_POR_DIAGNOSTICO[diagnostico]

    def tem_recurso_suficiente(self, acao):
        if acao == "Regar":
            return self.agua >= self.CUSTO_AGUA_POR_REGA and self.bateria >= self.CUSTO_BATERIA_POR_ACAO
        if acao == "Aplicar Defensivo":
            return self.defensivo >= self.CUSTO_DEFENSIVO_POR_APLICACAO and self.bateria >= self.CUSTO_BATERIA_POR_ACAO
        # "Ignorar" só custa bateria (o "custo cognitivo" de escanear/decidir)
        return self.bateria >= self.CUSTO_BATERIA_POR_ACAO

    def executar_acao(self, acao):
        """Consome os recursos correspondentes à ação. Assume que já se
        verificou tem_recurso_suficiente antes de chamar isso."""
        if acao == "Regar":
            self.agua -= self.CUSTO_AGUA_POR_REGA
        elif acao == "Aplicar Defensivo":
            self.defensivo -= self.CUSTO_DEFENSIVO_POR_APLICACAO
        self.bateria -= self.CUSTO_BATERIA_POR_ACAO

    # ---------- LOOP PRINCIPAL ----------
    def percorrer_estufa(self, estufa: Estufa, verboso=True):
        for (i, j) in estufa.posicoes():
            # custo de bateria só de se mover até a célula
            if self.bateria < self.CUSTO_BATERIA_POR_MOVIMENTO:
                self._registrar(
                    f"[PARADA] Bateria insuficiente para se mover até ({i},{j}). "
                    f"Agente retorna à base para recarregar.", verboso
                )
                break
            self.bateria -= self.CUSTO_BATERIA_POR_MOVIMENTO

            planta = estufa.get_planta(i, j)

            # PERCEPÇÃO
            leitura = self.perceber(planta)

            # RACIOCÍNIO
            diagnostico, probabilidades = self.raciocinar(leitura)
            planta.diagnostico_dado_agente = diagnostico
            planta.foi_visitada = True

            # comparação com a verdade (só para fins de análise/depuração -
            # o agente "de verdade" nunca teria acesso a isso em campo real)
            correto = (diagnostico == planta.estado_real)
            self.acertos += int(correto)
            self.erros += int(not correto)

            # AÇÃO
            acao = self.decidir_acao(diagnostico)

            if self.tem_recurso_suficiente(acao):
                self.executar_acao(acao)
                planta.acao_recebida = acao
                status = "OK" if correto else "OK(mas diagnostico != real)"
                self._registrar(
                    f"({i},{j}) sensores={leitura} -> diagnostico='{diagnostico}' "
                    f"(prob={probabilidades[diagnostico]:.2f}) real='{planta.estado_real}' "
                    f"[{status}] => acao='{acao}' | "
                    f"agua={self.agua} bateria={self.bateria} defensivo={self.defensivo}",
                    verboso,
                )
            else:
                planta.acao_recebida = "Pulada (recurso insuficiente)"
                self.acoes_puladas_por_falta_recurso += 1
                self._registrar(
                    f"({i},{j}) diagnostico='{diagnostico}' exigiria acao='{acao}' "
                    f"MAS recurso insuficiente (agua={self.agua}, "
                    f"defensivo={self.defensivo}, bateria={self.bateria}) "
                    f"-> acao racional: PULAR e seguir em frente.",
                    verboso,
                )

        return self.relatorio_final()

    def _registrar(self, mensagem, verboso):
        self.log.append(mensagem)
        if verboso:
            print(mensagem)

    def relatorio_final(self):
        total_diagnosticado = self.acertos + self.erros
        acuracia_campo = self.acertos / total_diagnosticado if total_diagnosticado else 0
        return {
            "acertos_diagnostico": self.acertos,
            "erros_diagnostico": self.erros,
            "acuracia_em_campo": acuracia_campo,
            "acoes_puladas_por_falta_recurso": self.acoes_puladas_por_falta_recurso,
            "agua_restante": self.agua,
            "bateria_restante": self.bateria,
            "defensivo_restante": self.defensivo,
        }