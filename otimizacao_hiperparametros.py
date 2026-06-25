"""
Script de Validação: Sintonia Fina de Hiperparâmetros em Pipeline Industrial.
Demonstra a compatibilidade nativa do NODORegulator com GridSearchCV.
"""

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from nodo_regulator import NODORegulator

# 1. Geração de base de dados de teste de alta entropia
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construção da estrutura do Pipeline Scikit-Learn
pipeline = Pipeline([
    ('nodo', NODORegulator()),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])

# 3. Definição da grade de busca de infraestrutura e modelagem
# O scikit-learn usa o prefixo 'nodo__' para acessar os parâmetros do nosso objeto
param_grid = {
    'nodo__n_components':,
    'nodo__k_neighbors': [10, 20, 30]
}

print("⚙️ Iniciando busca exaustiva de hiperparâmetros (GridSearchCV)...")

# 4. Execução da busca cruzada (Cross-Validation de 3 dobras)
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 5. Extração dos resultados homologados de engenharia
print("\n🏆 Processo de Otimização Concluído com Sucesso!")
print(f"🔹 Melhores parâmetros encontrados: {grid_search.best_params_}")
print(f"🔹 Melhor Score (F1-Weighted): {grid_search.best_score_:.4f}")

# 6. Avaliação final no conjunto de teste retido
test_score = grid_search.score(X_test, y_test)
print(f"🔹 Score de validação final nos dados de teste: {test_score:.4f}")
