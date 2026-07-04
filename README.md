sensores.py          -> gera leituras de sensores ruidosas a partir de um estado real
                         (usado tanto pelo dataset quanto pela estufa; garante que
                         treino e "produção" venham da mesma distribuição)

gerar_dataset.py      -> Passo 1: cria dataset_plantas.csv (~100 linhas históricas)

treinar_modelo.py     -> Passo 2: classe ModeloPlantas, que encapsula todo o
                         pré-processamento + GaussianNB do scikit-learn.
                         Expõe só `treinar()`, `prever()`, `salvar()`, `carregar()`.
                         >>> É A ÚNICA parte do código que conhece scikit-learn <<<

estufa.py             -> Passo 3: classes Planta e Estufa (o ambiente, uma matriz NxM)

agente.py             -> Passo 4: classe AgenteAgricultor (percepção -> raciocínio
                         via ModeloPlantas -> ação, com gestão de água/bateria/defensivo)
                         >>> NÃO importa scikit-learn, só chama modelo.prever(...) <<<

main.py               -> orquestra tudo (roda os 4 passos em sequência)

ANALISE.md            -> análise dos resultados (erros do Naive Bayes e
                         racionalidade do agente frente à escassez de recursos)

dataset_plantas.csv        -> dataset gerado (saída do Passo 1)
modelo_naive_bayes.pkl     -> modelo treinado e salvo (saída do Passo 2)
saida_execucao.txt         -> log completo de uma execução real do main.py