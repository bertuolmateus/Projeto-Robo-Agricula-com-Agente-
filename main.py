"""
main.py
-------
Orquestra o projeto do início ao fim:

  Passo 1 -> gerar_dataset.py     : cria dataset_plantas.csv
  Passo 2 -> treinar_modelo.py    : treina o Naive Bayes e mede acurácia
  Passo 3 -> estufa.py            : cria o ambiente (matriz 5x5 de plantas)
  Passo 4 -> agente.py            : agente percorre a estufa e age

Rode com: python main.py
"""

from gerar_dataset import gerar_dataset
from treinar_modelo import treinar_e_avaliar
from estufa import Estufa
from agente import AgenteAgricultor


def main():
    print("=" * 70)
    print("PASSO 1: Gerando dataset histórico de plantas...")
    print("=" * 70)
    gerar_dataset()

    print("\n" + "=" * 70)
    print("PASSO 2: Treinando o modelo Naive Bayes...")
    print("=" * 70)
    modelo = treinar_e_avaliar()

    print("\n" + "=" * 70)
    print("PASSO 3: Criando a estufa (ambiente 5x5)...")
    print("=" * 70)
    estufa = Estufa(linhas=5, colunas=5, seed=7)
    print("Distribuição real das plantas na estufa (verdade oculta ao robô):")
    print(estufa.resumo_estado_real())
    print("\nMapa da estufa (estado real, só para nós humanos vermos):")
    estufa.imprimir_mapa(mostrar_estado_real=True)

    print("\n" + "=" * 70)
    print("PASSO 4: Agente percorrendo a estufa...")
    print("=" * 70)
    # Recursos propositalmente limitados para forçar o agente a lidar com
    # escassez em algum momento da varredura (critério de análise de racionalidade)
    agente = AgenteAgricultor(modelo, agua_max=40, bateria_max=100, defensivo_max=15)
    relatorio = agente.percorrer_estufa(estufa, verboso=True)

    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL")
    print("=" * 70)
    for chave, valor in relatorio.items():
        print(f"  {chave}: {valor}")

    print("\nMapa final (diagnóstico dado pelo agente em cada célula):")
    estufa.imprimir_mapa(mostrar_estado_real=False)


if __name__ == "__main__":
    main()