import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

data = pd.read_csv("data.csv")

def print_class_distribution(df, title):
    print(f"\n{title}")
    print(df["category"].value_counts())

def balance_categories(df):
    max_count = df["category"].value_counts().max()
    balanced_parts = []

    for category, group in df.groupby("category"):
        # Upsample weaker classes so all classes have equal training samples.
        if len(group) < max_count:
            repeats = max_count // len(group)
            remainder = max_count % len(group)
            upsampled = pd.concat([group] * repeats + [group.sample(n=remainder, random_state=42)], ignore_index=True)
            balanced_parts.append(upsampled)
        else:
            balanced_parts.append(group)

    return pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

def evaluate_model(name, vectorizer, model, X_train, y_train, X_test, y_test, labels):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print("Confusion Matrix:")
    print(cm_df)

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": f1,
        "model": model,
        "vectorizer": vectorizer,
    }

print_class_distribution(data, "Original class distribution:")

balanced_data = balance_categories(data)
print_class_distribution(balanced_data, "Balanced class distribution:")

X = balanced_data["text"]
y = balanced_data["category"]
labels = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

baseline_result = evaluate_model(
    name="Baseline (TF-IDF unigram + Logistic Regression)",
    vectorizer=TfidfVectorizer(),
    model=LogisticRegression(max_iter=1000),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    labels=labels,
)

improved_result = evaluate_model(
    name="Improved (TF-IDF unigram+bigram + stopwords + class_weight)",
    vectorizer=TfidfVectorizer(ngram_range=(1, 2), stop_words="english"),
    model=LogisticRegression(max_iter=2000, class_weight="balanced"),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    labels=labels,
)

nb_result = evaluate_model(
    name="Alternative (TF-IDF unigram+bigram + stopwords + MultinomialNB)",
    vectorizer=TfidfVectorizer(ngram_range=(1, 2), stop_words="english"),
    model=MultinomialNB(alpha=0.5),
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    labels=labels,
)

best_result = max([baseline_result, improved_result, nb_result], key=lambda x: x["macro_f1"])

print("\nModel comparison summary:")
print(f"- Baseline Macro F1: {baseline_result['macro_f1']:.4f}")
print(f"- Improved LR Macro F1: {improved_result['macro_f1']:.4f}")
print(f"- MultinomialNB Macro F1: {nb_result['macro_f1']:.4f}")
print(f"Recommended model: {best_result['name']}")

# Save the best-performing model for app.py inference.
joblib.dump(best_result["model"], "model.pkl")
joblib.dump(best_result["vectorizer"], "vectorizer.pkl")

print("\nSaved best model and vectorizer to model.pkl and vectorizer.pkl")

client_history = []

while True:
    complaint = input("Enter complaint (or type exit): ")
    if complaint.lower() == "exit":
        break

    location = input("Enter location: ")

    combined = complaint + " " + location
    text_vec = best_result["vectorizer"].transform([combined])
    prediction = best_result["model"].predict(text_vec)
    print("Predicted category:", prediction[0])

    client_history.append({"location": location, "category": prediction[0]})

    if client_history:
        history_df = pd.DataFrame(client_history)
        print("\nRegion-wise issue distribution from client submissions:")
        print(history_df.groupby("location")["category"].value_counts())

    analyze = input("Analyze most common issue for this location? (yes/no): ").strip().lower()
    if analyze == "yes":
        history_df = pd.DataFrame(client_history)
        subset = history_df[history_df["location"].str.lower() == location.lower()]
        if subset.empty:
            print("No historical records available for this location.")
        else:
            print("Most common issue:", subset["category"].value_counts().idxmax())

