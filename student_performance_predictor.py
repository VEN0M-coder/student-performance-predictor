# ============================================================
#   STUDENT PERFORMANCE PREDICTOR
#   ML Project by Divyansh Rai | B.Tech IT, GEC Ajmer
#   Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   STUDENT PERFORMANCE PREDICTOR")
print("=" * 55)

# ── STEP 2: Load Dataset ──────────────────────────────────────
df = pd.read_csv('student_data.csv')

print("\n📦 Dataset loaded successfully!")
print(f"   Shape: {df.shape[0]} students, {df.shape[1]} features")
print("\n🔍 First 5 rows:")
print(df.head())

print("\n📊 Basic Statistics:")
print(df.describe().round(2))

print("\n✅ Missing values:", df.isnull().sum().sum(), "(none)")

# ── STEP 3: Exploratory Data Analysis (EDA) ──────────────────
print("\n📈 Generating EDA charts...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Student Performance - Exploratory Data Analysis',
             fontsize=16, fontweight='bold')

# 1. Pass/Fail distribution
labels = ['Fail', 'Pass']
counts = df['result'].value_counts().sort_index()
colors = ['#e74c3c', '#2ecc71']
axes[0, 0].bar(labels, counts, color=colors, edgecolor='black', width=0.5)
axes[0, 0].set_title('Pass / Fail Distribution')
axes[0, 0].set_ylabel('Number of Students')
for i, v in enumerate(counts):
    axes[0, 0].text(i, v + 3, str(v), ha='center', fontweight='bold')

# 2. Study Hours vs Result
df.boxplot(column='study_hours', by='result', ax=axes[0, 1],
           patch_artist=True)
axes[0, 1].set_title('Study Hours vs Result')
axes[0, 1].set_xlabel('Result (0=Fail, 1=Pass)')
axes[0, 1].set_ylabel('Study Hours')
plt.sca(axes[0, 1])
plt.title('Study Hours vs Result')

# 3. Attendance vs Result
df.boxplot(column='attendance_percent', by='result', ax=axes[0, 2],
           patch_artist=True)
axes[0, 2].set_title('Attendance % vs Result')
axes[0, 2].set_xlabel('Result (0=Fail, 1=Pass)')
plt.sca(axes[0, 2])
plt.title('Attendance % vs Result')

# 4. Previous Score distribution
axes[1, 0].hist(df[df['result'] == 1]['previous_score'],
                bins=20, alpha=0.7, label='Pass', color='#2ecc71')
axes[1, 0].hist(df[df['result'] == 0]['previous_score'],
                bins=20, alpha=0.7, label='Fail', color='#e74c3c')
axes[1, 0].set_title('Previous Score Distribution')
axes[1, 0].set_xlabel('Score')
axes[1, 0].legend()

# 5. Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title('Feature Correlation Heatmap')

# 6. Study Hours distribution
axes[1, 2].hist(df['study_hours'], bins=20,
                color='#3498db', edgecolor='black')
axes[1, 2].set_title('Study Hours Distribution')
axes[1, 2].set_xlabel('Hours')
axes[1, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ EDA charts saved → eda_charts.png")

# ── STEP 4: Prepare Data for ML ───────────────────────────────
print("\n⚙️  Preparing data for ML...")

# Features (X) and Target (y)
X = df.drop('result', axis=1)   # input features
y = df['result']                 # what we want to predict

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── STEP 5: Train Models ──────────────────────────────────────
print("\n🤖 Training ML Models...")

# Model 1: Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1])

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"\n   Logistic Regression → Accuracy: {lr_acc*100:.2f}%  AUC: {lr_auc:.3f}")
print(f"   Random Forest       → Accuracy: {rf_acc*100:.2f}%  AUC: {rf_auc:.3f}")

best_model = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\n   🏆 Best Model: {best_model}")

# ── STEP 6: Evaluate Best Model ───────────────────────────────
print("\n📋 Classification Report (Random Forest):")
print(classification_report(y_test, rf_pred,
                             target_names=['Fail', 'Pass']))

# ── STEP 7: Visualize Results ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Evaluation Results', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
axes[0].set_title('Confusion Matrix\n(Random Forest)')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Model Comparison
models = ['Logistic\nRegression', 'Random\nForest']
accs   = [lr_acc * 100, rf_acc * 100]
bars = axes[1].bar(models, accs, color=['#3498db', '#2ecc71'],
                   edgecolor='black', width=0.4)
axes[1].set_title('Model Accuracy Comparison')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim([0, 110])
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', fontweight='bold')

# Feature Importance (Random Forest)
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', ax=axes[2],
                             color='#9b59b6', edgecolor='black')
axes[2].set_title('Feature Importance\n(Random Forest)')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Model result charts saved → model_results.png")

# ── STEP 8: Predict on New Student ───────────────────────────
print("\n" + "=" * 55)
print("   🎓 PREDICT FOR A NEW STUDENT")
print("=" * 55)

new_student = pd.DataFrame({
    'study_hours':              [7.0],
    'attendance_percent':       [85],
    'previous_score':           [72],
    'sleep_hours':              [7.0],
    'extracurricular_activities': [1]
})

prediction = rf.predict(new_student)[0]
probability = rf.predict_proba(new_student)[0]

print(f"\n   Input:")
print(f"   Study Hours       : 7.0 hrs/day")
print(f"   Attendance        : 85%")
print(f"   Previous Score    : 72")
print(f"   Sleep Hours       : 7.0 hrs")
print(f"   Extracurricular   : Yes")
print(f"\n   Prediction  : {'✅ PASS' if prediction == 1 else '❌ FAIL'}")
print(f"   Confidence  : {max(probability)*100:.1f}%")
print(f"   Pass Probability  : {probability[1]*100:.1f}%")
print(f"   Fail Probability  : {probability[0]*100:.1f}%")

print("\n" + "=" * 55)
print("   PROJECT COMPLETE ✅")
print("   Files generated:")
print("   → eda_charts.png   (data analysis charts)")
print("   → model_results.png (ML model results)")
print("=" * 55)
