"""
Passo 2: Treinamento do Modelo (Machine Learning)
--------------------------------------------------
Carrega o dataset_plantas.csv com pandas, treina um classificador Naive Bayes
(GaussianNB) e mede sua acurácia num conjunto de teste separado.

Por que GaussianNB e não CategoricalNB?
    Nosso conjunto de features é MISTO: 'cor_folha' e 'tamanho' são
    categóricas, enquanto 'umidade_solo' é contínua (%). A abordagem mais
    simples e didática é converter as categóricas em códigos numéricos
    (LabelEncoder) e usar GaussianNB, que assume que cada feature, dentro de
    cada classe, segue uma distribuição normal. É uma simplificação (uma
    variável categórica não é "gaussiana" de fato), mas funciona bem na
    prática para poucas categorias e é o que se costuma ensinar em cursos
    introdutórios de IA. (Uma alternativa mais "correta" seria usar
    CategoricalNB só para cor_folha/tamanho e discretizar a umidade em
    faixas - fica como sugestão de extensão do projeto.)

Este módulo expõe a função `treinar_e_avaliar()` que:
  1. Carrega o CSV.
  2. Codifica as colunas categóricas.
  3. Separa treino/teste (80/20).
  4. Treina o GaussianNB.
  5. Calcula acurácia e matriz de confusão.
  6. Salva o modelo + encoders em modelo_naive_bayes.pkl (via pickle).

O agente (agente.py) NUNCA lida com scikit-learn diretamente: ele só chama
`prever(leitura_sensores)` da classe ModeloPlantas abaixo. Isso mantém a
lógica de ML separada da lógica do agente (requisito de arquitetura).
"""

import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ARQUIVO_DATASET = "dataset_plantas.csv"
ARQUIVO_MODELO = "modelo_naive_bayes.pkl"


class ModeloPlantas:
    """
    Encapsula o modelo Naive Bayes + os encoders das colunas categóricas.
    Fornece uma API simples (`prever`) para o agente usar, escondendo todos
    os detalhes de pré-processamento e do scikit-learn.
    """

    def __init__(self):
        self.modelo = GaussianNB()
        self.encoder_cor = LabelEncoder()
        self.encoder_tamanho = LabelEncoder()
        self.encoder_diagnostico = LabelEncoder()
        self.acuracia_teste = None

    def _preparar_features(self, df: pd.DataFrame, treinar_encoders=False) -> pd.DataFrame:
        df = df.copy()
        if treinar_encoders:
            df["cor_folha_cod"] = self.encoder_cor.fit_transform(df["cor_folha"])
            df["tamanho_cod"] = self.encoder_tamanho.fit_transform(df["tamanho"])
        else:
            df["cor_folha_cod"] = self.encoder_cor.transform(df["cor_folha"])
            df["tamanho_cod"] = self.encoder_tamanho.transform(df["tamanho"])
        return df[["cor_folha_cod", "umidade_solo", "tamanho_cod"]]

    def treinar(self, caminho_csv=ARQUIVO_DATASET):
        df = pd.read_csv(caminho_csv)

        X = self._preparar_features(df, treinar_encoders=True)
        y = self.encoder_diagnostico.fit_transform(df["diagnostico"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.modelo.fit(X_train, y_train)

        y_pred = self.modelo.predict(X_test)
        self.acuracia_teste = accuracy_score(y_test, y_pred)

        print(f"[OK] Modelo treinado com {len(X_train)} amostras "
              f"(teste com {len(X_test)} amostras)")
        print(f"[OK] Acurácia no conjunto de teste: {self.acuracia_teste:.2%}")
        print("\nMatriz de confusão (linhas=real, colunas=previsto):")
        classes = self.encoder_diagnostico.classes_
        cm = confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(cm, index=classes, columns=classes))
        print("\nRelatório de classificação:")
        print(classification_report(
            y_test, y_pred, target_names=classes, zero_division=0
        ))

        return self.acuracia_teste

    def prever(self, leitura_sensores: dict):
        """
        leitura_sensores: dict com chaves 'cor_folha', 'umidade_solo', 'tamanho'
        Retorna: (diagnostico_previsto: str, probabilidades: dict{classe: prob})
        """
        df = pd.DataFrame([leitura_sensores])
        X = self._preparar_features(df, treinar_encoders=False)

        pred_cod = self.modelo.predict(X)[0]
        probas = self.modelo.predict_proba(X)[0]

        diagnostico = self.encoder_diagnostico.inverse_transform([pred_cod])[0]
        dict_probas = {
            classe: prob
            for classe, prob in zip(self.encoder_diagnostico.classes_, probas)
        }
        return diagnostico, dict_probas

    def salvar(self, caminho=ARQUIVO_MODELO):
        with open(caminho, "wb") as f:
            pickle.dump(self, f)
        print(f"[OK] Modelo salvo em: {caminho}")

    @staticmethod
    def carregar(caminho=ARQUIVO_MODELO):
        with open(caminho, "rb") as f:
            return pickle.load(f)


def treinar_e_avaliar():
    modelo = ModeloPlantas()
    modelo.treinar()
    modelo.salvar()
    return modelo


if __name__ == "__main__":
    treinar_e_avaliar()