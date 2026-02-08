# Public Grievance Classification Dashboard

This project is an NLP-based public grievance classification system that automatically categorizes citizen complaints into relevant departments such as " Electricity, Water, Roads, and Sanitation ".

The machine learning model is deployed as a lightweight " Flask web dashboard ", enabling real-time complaint classification and maintaining a session-based history of predictions. 
The project demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment and version control.

# Features :

- Automatic text classification of public grievances  
- NLP pipeline using " TF-IDF vectorization "  
- Classification using " Logistic Regression "
- Real-time prediction through a Flask web dashboard  
- Session-based prediction history table  
- Simple, interpretable, and explainable ML approach  
- Version-controlled and published on GitHub  


# 🛠️ Tech Stack :

-  Programming Language: Python  
-  Machine Learning: Scikit-learn  
-  Data Processing: Pandas, NumPy  
-  Web Framework: Flask  
-  Frontend: HTML, CSS  
-  Version Control: Git & GitHub  

# How to Run Locally :

Follow the steps below to run this project on your local system.

# 1 . Clone the repository

```bash
git clone https://github.com/Nirupama1009/grievance-classification-dashboard.git
cd grievance-classification-dashboard
```

# 2️. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

# 3️ . Install required dependencies

```bash
pip install pandas numpy scikit-learn flask nltk matplotlib joblib
```

# 4️ . Train the machine learning model

```bash
python train_model.py
```

This step generates the trained model and vectorizer files used by the dashboard.

---

# 5️ . Run the Flask web application

```bash
python app.py
```

# 6️ . Open the dashboard in your browser

```
http://127.0.0.1:5000
```

# Notes & Limitations :

- The dataset used is limited and intended for " demonstration purposes ".  
- The prediction history is " session-based " and resets when the page is refreshed.  
- The application runs on Flask’s development server and is " not intended for production deployment ".  


# Future Enhancements :

- Integration with a database for persistent complaint history  
- Support for larger, real-world datasets  
- Improved evaluation with cross-validation  
- Deployment to a cloud platform  

# Author :

Nirupama R 
Computer Science (AIML) Student  
GitHub: https://github.com/Nirupama1009  



  # " Acknowledgment "

This project was developed as a learning initiative to understand applied NLP systems and real-world deployment of machine learning models using Python and Flask.
