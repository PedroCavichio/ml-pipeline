#train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Leitura dos dados
df = pd.read_csv("data/sample.csv")

# 2. Separar features e target
X = df.drop("target", axis=1)
y = df["target"]

# 3. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Treinamento
model = LogisticRegression(max_iter=1000)  # previne warnings de convergência
model.fit(X_train, y_train)

# 5. Avaliação
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# 6. Salvamento do relatório
with open("report.txt", "w") as f:
    f.write(report)

print("Relatório gerado em report.txt")
