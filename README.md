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
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn  
- **Database**: SQLite  
- **Notebook**: Jupyter Notebook  

---

## **Results**  
- Achieved a **recall score of 65.00**, reducing the risk of undetected diabetes cases.   I plan too address this by adjusting the model threshold, undersampling the majority
  cases and adding a 'balanced' class weight.

- Identified key health metrics strongly correlated with diabetes risk.  

---

 
