# 📊 Student Performance Predictor

A Machine Learning project that predicts whether a student will **Pass or Fail** based on study habits and academic data — built with Python, Scikit-learn, Pandas, and Matplotlib.

---

## 🎯 What It Does

Takes 5 input features about a student and predicts their result with confidence score:

| Input Feature | Example |
|---|---|
| Study Hours per day | 7.0 hrs |
| Attendance % | 85% |
| Previous Score | 72 |
| Sleep Hours per day | 7.0 hrs |
| Extracurricular Activities | Yes / No |

**Output:** ✅ PASS or ❌ FAIL with probability score

---

## 🤖 Models Used

| Model | Accuracy | AUC Score |
|---|---|---|
| Logistic Regression | 100% | 1.000 |
| Random Forest | 93% | 0.992 |

> Best model selected automatically based on accuracy

---

## 📈 What's Inside

### 1. Exploratory Data Analysis (EDA)
- Pass/Fail distribution bar chart
- Study hours vs Result boxplot
- Attendance % vs Result boxplot
- Previous score distribution (Pass vs Fail overlay)
- Feature correlation heatmap
- Study hours histogram

### 2. ML Pipeline
- Data preprocessing with `StandardScaler`
- 80/20 train/test split with stratification
- Training 2 models: Logistic Regression + Random Forest
- Evaluation: Accuracy, AUC-ROC, Confusion Matrix, Classification Report

### 3. Feature Importance
Identifies which factors affect student performance the most using Random Forest feature importance scores.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | ML models & evaluation |
| Matplotlib | Charts & plots |
| Seaborn | Heatmaps & styled charts |

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/VEN0M-coder/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the project
```bash
python student_performance_predictor.py
```

### Output files generated:
- `eda_charts.png` — data analysis visualizations
- `model_results.png` — confusion matrix, accuracy comparison, feature importance

---

## 📁 Project Structure

```
student-performance-predictor/
│
├── student_performance_predictor.py   # Main ML script
├── student_data.csv                   # Dataset (500 students)
├── eda_charts.png                     # EDA visualizations (auto-generated)
├── model_results.png                  # Model results (auto-generated)
└── README.md
```

---

## 🔍 Sample Prediction

```
Input:
  Study Hours       : 7.0 hrs/day
  Attendance        : 85%
  Previous Score    : 72
  Sleep Hours       : 7.0 hrs
  Extracurricular   : Yes

Output:
  Prediction  : ✅ PASS
  Confidence  : 100.0%
  Pass Probability : 100.0%
  Fail Probability : 0.0%
```

---

## 📊 Results Preview

**EDA Charts** — distribution and correlation analysis across all features

**Model Results** — confusion matrix, model comparison, and feature importance ranking

> Top predictors: Study Hours and Previous Score have the highest impact on pass/fail outcome

---

## 👨‍💻 Author

**Divyansh Rai**
B.Tech Information Technology — Government Engineering College, Ajmer
📧 divyansharye@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/divyansh-rai-2763a4353/)
🐙 [GitHub](https://github.com/VEN0M-coder)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
