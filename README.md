
<img width="750" alt="Screenshot 2024-12-02 at 9 30 46 PM" src="https://github.com/user-attachments/assets/0971f968-5432-4d1c-aad9-16d745a5c663">



# **Healthcare Analytics Project**  
*Using Machine Learning to Predict Diabetes Risk* 

---

## **Overview**  
This project leverages a Kaggle healthcare dataset to analyze patient health metrics and predict diabetes risk. The analysis includes data preprocessing, exploratory data analysis (EDA), machine learning modeling, and visualization to provide actionable insights for healthcare decision-making.

---

## **Dataset**  
- **Source**: [Kaggle Healthcare Dataset](https://www.kaggle.com/)  
- **Description**: Contains patient health metrics (e.g., glucose levels, BMI, blood pressure) and a target variable indicating diabetes presence.  

---

## **Objectives**  
1. **Analyze** patient health data to uncover patterns and correlations.  
2. **Build** a machine learning model for diabetes risk prediction.  
3. **Optimize** the model for high recall to minimize false negatives.  
4. **Visualize** findings to support data-driven healthcare decisions.

---

## **Project Highlights**  

### 🔹 **Data Preprocessing**  
- Addressed missing values and cleaned data inconsistencies.  
- Transformed and scaled features for improved modeling performance.  

### 🔹 **Exploratory Data Analysis (EDA)**  
- Visualized feature relationships with correlation heatmaps, boxplots, and histograms.  
- Identified key metrics like glucose levels and BMI impacting diabetes risk.  

### 🔹 **Machine Learning Model**  
- Trained a `RandomForestClassifier` for diabetes risk prediction.  
- Adjusted decision thresholds to prioritize recall and reduce false negatives.  

### 🔹 **Visualization**  
- Communicated findings with insightful visualizations to support healthcare professionals.

---

## **Tools and Technologies**  
- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Imblearn  
- **Notebook**: Google Colab Notebook  

---

## **Results**  
Initial Model Performance

The initial model was a RandomForestClassifier trained with class weighting (class_weight='balanced'). The recall score was 0.65, indicating the model correctly identified 65% of positive cases (diabetes). This was a decent starting point but left room for improvement to reduce false negatives further.

Threshold Adjustment

By lowering the decision threshold from the default (0.5) to 0.3, the model prioritized sensitivity to detect more positive cases. This adjustment improved the recall score to 0.87, ensuring more positive cases were identified. However, this came at the cost of reduced precision, as more false positives were introduced.

SMOTE (Synthetic Minority Oversampling Technique)

To address class imbalance, SMOTE was applied to oversample the minority class (positive cases). After retraining the model, the recall score improved to 0.78, while maintaining a better balance between precision and recall. This method showed significant improvement without drastically sacrificing precision.

Grid Search for Hyperparameter Tuning

To optimize the model further, hyperparameter tuning was conducted using GridSearchCV. This step fine-tuned parameters like n_estimators, maz_depth, and min_smaple_split. Post-tuning, the recall score remained 0.78, indicating stability, but the f1-score improved slightly, showing better overall performance.

---

 
