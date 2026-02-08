import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("data.csv")

X = data["text"]
y = data["category"]

X_train, X_test, y_train, y_test = X, X, y, y

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, labels=y.unique(), zero_division=0))

# Live prediction
while True:
    text = input("Enter a complaint (or type exit): ")
    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    print("Predicted category:", prediction[0])

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")

