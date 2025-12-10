
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

 # **Healthcare Analytics Project**  
*Predicting Diabetes Risk Using Machine Learning*

---

## **Overview**
This project analyzes patient health metrics to predict diabetes risk using a modular, production-style machine learning pipeline.  
The workflow includes:

- Data preprocessing and cleaning  
- Exploratory data analysis (EDA)  
- Model training using a recall-optimized Random Forest  
- Standardized model evaluation  
- Optional command-line prediction generation  

The objective is to demonstrate an analytics engineering approach that is **reproducible, interpretable, and clinically meaningful**, especially for use cases where reducing false negatives matters.

---

## **Dataset**
**Source:** Public diabetes dataset inspired by the Pima Indians dataset  
**Description:** Includes features such as glucose, BMI, blood pressure, insulin, age, diabetes pedigree function, and an outcome label indicating diabetes status.

---

## **Objectives**
1. Explore patient health metrics and identify meaningful predictors.  
2. Build an ML classifier optimized for **high recall** (minimizing false negatives).  
3. Use a modular, production-inspired project structure.  
4. Provide clear evaluation metrics and interpretable insights.  
5. Develop a scalable baseline for future healthcare analytics workflows.

---

## **Project Highlights**

### **1. Data Preprocessing**
- Converted non-physiological zero values to missing values  
- Applied median imputation across relevant fields  
- Centralized all cleaning logic in `src/data/clean_data.py`  

### **2. Exploratory Data Analysis (EDA)**
- Investigated variable distributions, correlations, and feature interactions  
- Identified glucose, BMI, age, and pedigree function as key predictors  
- Restricted notebooks to exploratory purposes only; production logic is modular  

### **3. Machine Learning Model**
- Used a `RandomForestClassifier` with class weighting to address imbalance  
- Performed hyperparameter tuning via GridSearchCV (scoring = recall)  
- Implemented training and evaluation pipelines in `src/models/`  
- Saved the final trained model to the `models/` directory  

### **4. Visualization**
Tools provided in `src/visualization/` include:

- ROC curve generation  
- Feature importance visualization  
- Confusion matrix plotting  

---

## **Tools and Technologies**
- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- joblib  
- argparse  
- Jupyter Notebook  
- Modular analytics engineering structure

---

## **Results**

### **Final Model Performance (Hold-Out Test Set)**
- Accuracy: 0.747  
- Precision: 0.632  
- Recall: 0.667  
- F1 Score: 0.649  
- ROC AUC: 0.821  

### **Interpretation**
- **Recall (0.667):** The model captures most true diabetes cases, aligning with the priority of reducing false negatives.  
- **ROC AUC (0.821):** Indicates strong class separability and reliable predictive discrimination.  
- Precision and F1 remain balanced while recall is prioritized.

Earlier experimental approaches—threshold shifting, SMOTE oversampling, and initial hyperparameter tuning—helped guide the final implementation.  
The structured pipeline improves clarity, reproducibility, and engineering quality without sacrificing clinical relevance.

---

## **Pipeline Usage**

### **Training the Model**
Run the following command:

    python main.py train

This handles data cleaning, splitting, hyperparameter tuning, model evaluation, and saving the final model.

### **Generating Predictions**
To create predictions:

    python main.py predict --input data/raw/diabetes.csv --output predictions.csv

The output file will contain a new column named `prediction`.

---

## **Future Improvements**
- Decision threshold tuning for recall/precision optimization  
- Optional SMOTE or alternative class balancing  
- SHAP for interpretability  
- FastAPI for real-time scoring  
- MLflow for experiment tracking  
- Unit testing and CI/CD pipeline integration  

---

## **Author**
**Julian Charlan Kelly**  
Analytics Engineer / Data Engineer  
Los Angeles, CA
