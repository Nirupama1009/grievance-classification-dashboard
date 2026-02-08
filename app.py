from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

history = []
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    complaint_text = None

    if request.method == "POST":
        complaint_text = request.form["complaint"]
        complaint_vec = vectorizer.transform([complaint_text])
        prediction = model.predict(complaint_vec)[0]

        history.append({
            "complaint": complaint_text,
            "prediction": prediction
        })

    return render_template(
        "index.html",
        prediction=prediction,
        history=history
    )

if __name__ == "__main__":
    app.run(debug=True)
