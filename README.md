# 🚢 Titanic Survival Prediction using Machine Learning Pipeline

This project demonstrates the use of **Machine Learning Pipelines** in Python using **scikit-learn** to predict survival on the Titanic dataset. The pipeline handles all preprocessing steps and applies a classifier in a clean, reproducible way.

---

## 📁 Project Structure

```
├── Machine pipeline.ipynb             # Main notebook with model building pipeline
├── predict using pipeline.ipynb       # Notebook to use trained pipeline for predictions
├── pipe.pkl                           # Trained pipeline saved as a pickle file
├── README.md                          # Project documentation
```

---

## 🚀 Features

- Complete ML pipeline including preprocessing and model training
- Handling missing values and categorical encoding
- Pipeline serialization using `joblib`
- Inference using the saved pipeline
- Simple and extendable structure

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

You’ll need:

- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`
- `matplotlib` (optional for visualizations)

---

## 📊 Dataset

The dataset used is the classic [Titanic dataset](https://www.kaggle.com/c/titanic/data).  
It includes features such as `Pclass`, `Sex`, `Age`, `Fare`, and survival labels (`Survived`).

---

## 🧠 Model Pipeline

The pipeline includes the following steps:

1. **Imputation**: Filling missing values (e.g., age, embarked).
2. **Encoding**: Converting categorical variables (`Sex`, `Embarked`) using OneHotEncoding.
3. **Feature Scaling**: StandardScaler for numeric features.
4. **Feature Selection**: (Optional) using `SelectKBest`.
5. **Classification**: Using `RandomForestClassifier`.

---

## 🛠 How to Use

1. **Train the model:**
   Open `Machine pipeline.ipynb` and run all cells. This notebook creates the pipeline, trains it, and saves it to `pipe.pkl`.

2. **Predict using the saved model:**
   Open `predict using pipeline.ipynb` to load the trained model and make predictions on new or test data.

---

## 🔍 Example Prediction

```python
import joblib
import pandas as pd

pipe = joblib.load("pipe.pkl")
new_data = pd.DataFrame([{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "Parch": 0,
    "Embarked": "S"
}])
prediction = pipe.predict(new_data)
print("Survived" if prediction[0] == 1 else "Did not survive")
```

---

## 📚 Learn More

- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

---

## 🙌 Acknowledgements

- Kaggle for the dataset.
- scikit-learn for the pipeline and modeling tools.
