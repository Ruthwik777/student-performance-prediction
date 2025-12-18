# student-performance-prediction
# Student Performance Prediction using Machine Learning

## 1. Abstract
This project aims to predict student academic performance using simple and widely accepted machine learning techniques. Based on student-related attributes such as attendance percentage, internal assessment marks, assignment score, previous semester marks, and study hours per week, the system predicts an academic grade category (A, B, C, or D). Two standard classification algorithms—Logistic Regression and Decision Tree Classifier—are trained and evaluated. The project includes data preprocessing, model training, evaluation using common metrics, and visualizations suitable for academic demonstration and faculty evaluation.

## 2. Problem Statement
Educational institutions often need early indicators of student performance to provide timely academic support. The problem addressed in this project is:
To build a machine learning model that can predict a student’s performance category (Grade A/B/C/D) using basic academic and behavioral input features.

## 3. Dataset Description
A sample dataset is used in CSV format: `dataset/student_data.csv`.

### Input Features
- AttendancePercentage
- InternalAssessmentMarks
- AssignmentScore
- PreviousSemesterMarks
- StudyHoursPerWeek

### Output Label
- Grade (A, B, C, D)

The dataset includes a few missing values to demonstrate missing-value handling during preprocessing.

## 4. Algorithms Used
1. Logistic Regression  
   - A simple and standard linear model for classification.
2. Decision Tree Classifier  
   - A widely used and interpretable model that can show feature importance.

## 5. System Requirements
### Software
- Python 3.9+ (recommended)
- VS Code (or any Python IDE)

### Python Libraries
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib

Install dependencies using:
pip install -r requirements.txt

## 6. Steps to Run the Project
1. Open the folder `student-performance-prediction/` in VS Code.
2. (Recommended) Create and activate a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate
3. Install requirements:
   - pip install -r requirements.txt
4. Run the project:
   - python main.py

## 7. Results and Discussion
The project evaluates both Logistic Regression and Decision Tree models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Generated outputs:
- `results/confusion_matrix.png`  
  Confusion matrices for both models to visualize correct vs incorrect classifications.
- `results/accuracy_comparison.png`  
  A bar chart comparing model accuracies.
- `results/feature_importance.png`  
  Feature importance plot for the Decision Tree model.

In general:
- Logistic Regression provides a simple baseline.
- Decision Tree offers interpretability and feature importance, which is useful for academic analysis.

## 8. Conclusion
This project demonstrates a complete machine learning workflow for student performance prediction using standard classification algorithms. The implemented pipeline covers dataset loading, preprocessing, training, evaluation, and result visualization. Such a system can be extended to help institutions identify students who may need additional support.

## 9. Future Scope
- Collect a larger real-world dataset and improve generalization.
- Add cross-validation for more reliable evaluation.
- Try additional standard algorithms such as KNN or Naive Bayes.
- Extend output to both Grade (A/B/C/D) and Pass/Fail simultaneously.
- Build a simple GUI or web interface for user-friendly predictions.

## 10. Repository Structure
student-performance-prediction/
- dataset/ : CSV dataset
- src/ : preprocessing, training, evaluation modules
- models/ : saved trained model pickle
- results/ : saved plots
- main.py : end-to-end runner script
  
