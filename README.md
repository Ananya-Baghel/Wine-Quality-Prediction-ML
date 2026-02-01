# 🍷 Wine Quality Prediction using Machine Learning

This project is a basic machine learning application that predicts the quality of red wine using its physicochemical properties.  
A **Random Forest Classifier** is used to train the model and make predictions based on the given dataset.

The project is implemented and executed in **Google Colab** and later structured for hosting on **GitHub**.

---

## 📌 Project Description

Wine quality depends on various chemical characteristics such as acidity, alcohol content, sulphates, and pH value.  
This project analyzes these features and builds a machine learning model to predict wine quality.

The focus of this project is:
- Understanding the dataset
- Performing basic data analysis
- Training a Random Forest model
- Evaluating model performance

This is a **beginner-friendly ML project** suitable for academic submission and portfolio building.

---

## 📂 Dataset Details

- **Dataset Name:** Red Wine Quality Dataset  
- **Source:** UCI Machine Learning Repository  
- **File Used:** `winequality-red.csv`  
- **Target Column:** `quality`  

### Input Features:
- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

---

## 🛠️ Tools & Technologies Used

- Python  
- Google Colab  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🔍 Exploratory Data Analysis (EDA)

The following steps were performed:
- Checked dataset shape and basic information
- Verified missing values
- Visualized feature correlations using a heatmap
- Analyzed how different features affect wine quality

---

## 🤖 Machine Learning Model

### Model Used:
- **Random Forest Classifier**

### Workflow:
1. Data preprocessing
2. Feature selection
3. Train-test split
4. Model training using Random Forest
5. Prediction on test data
6. Model evaluation

---

## 📊 Model Evaluation

The model was evaluated using:
- Accuracy Score
- Classification performance metrics

The Random Forest model provided satisfactory accuracy for a basic classification task and performed better than simple baseline models.

---

## 🚀 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/Ananya-Baghel/Wine-Quality-Prediction-ML.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the notebook:
```bash
wine_quality_prediction.ipynb
```

---

## 📁 Project Structure

```
Wine-Quality-Prediction-ML/
│
├── wine_quality_prediction.ipynb
├── winequality-red.csv
├── requirements.txt
└── README.md
```

---

## 📈 Key Insights

- Alcohol content plays a major role in determining wine quality
- Volatile acidity negatively impacts wine quality
- Random Forest provides reliable performance for this dataset

---

## ✨ Future Scope

- Hyperparameter tuning of Random Forest
- Trying other classifiers (SVM, Gradient Boosting)
- Deploying the model using Streamlit or Flask
- Improving feature engineering

---

## 👩‍💻 Author

**Ananya Baghel**  
Machine Learning Enthusiast  

---

⭐ This project is intended for learning and academic purposes.
