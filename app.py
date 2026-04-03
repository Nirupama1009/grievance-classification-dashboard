from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

history = list()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    complaint_text = ""
    location_text = ""
    error_message = None

    if request.method == "POST":
        complaint_text = request.form.get("complaint", "").strip()
        location_text = request.form.get("location", "").strip()

        if not complaint_text or not location_text:
            error_message = "Please enter both complaint and location."
            return render_template(
                "index.html",
                prediction=prediction,
                complaint_text=complaint_text,
                location_text=location_text,
                error_message=error_message,
                history=history,
            )

        combined_text = complaint_text + " " + location_text

        complaint_vec = vectorizer.transform([combined_text])
        prediction = model.predict(complaint_vec)[0]

        history.append({
            "complaint": complaint_text,
            "location": location_text,
            "prediction": prediction,
        })

    return render_template(
        "index.html",
        prediction=prediction,
        complaint_text=complaint_text,
        location_text=location_text,
        error_message=error_message,
        history=history,
    )

if __name__ == "__main__":
    app.run(debug=True)
