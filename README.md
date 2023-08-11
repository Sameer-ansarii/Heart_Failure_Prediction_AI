# Heart_Failure_Prediction_AI
![giphy](https://github.com/Sameer-ansarii/Heart_Failure_Prediction_AI/assets/125865393/4eeadd93-e4c7-4ee7-ae63-10d83507b0ff)

Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Heart failure is a common event caused by CVDs, and this dataset contains 12 features that can be used to predict mortality from heart failure.

Most CVDs can be prevented by addressing behavioral risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity, and harmful use of alcohol. These can be addressed through population-wide strategies.

People with CVD or who are at high CVD risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management. Machine learning models can be of great help in this regard.

​

# Problem Statement

Heart failure is a critical medical condition that affects millions of people worldwide. It is a major cause of death and disability, and it can have a significant impact on a person's quality of life. Early detection and prediction of heart failure can play a crucial role in providing timely medical intervention and improving patient outcomes.

In this project, our goal is to develop a predictive model that can accurately identify individuals at risk of heart failure based on a set of clinical and demographic features. By analyzing medical records and relevant parameters, we aim to create a reliable tool that assists healthcare professionals in identifying patients who require closer monitoring and tailored treatment plans.

The primary objective of this project is to harness the power of data science and machine learning to enhance early detection of heart failure. This could potentially save lives and improve the quality of healthcare for individuals susceptible to this condition.

# Data Set Information

* **age**:  Age of the patient
​
​
* **anaemia**: Haemoglobin level of patient (Boolean)
​
​
* **eatinine_phosphokinase**Level of the CPK enzyme in the blood (mcg/L)
​
​
* **diabetes**: If the patient has diabetes (Boolean)
​
​
* **ejection_fraction**: Percentage of blood leaving the heart at each contraction
​
​
* **high_blood_pressure**: If the patient has hypertension (Boolean)
​
​
* **platelets**: Platelet count of blood (kiloplatelets/mL)
​
​
* **serum_creatinine**: Level of serum creatinine in the blood (mg/dL)
​
​
* **serum_sodium**: Level of serum sodium in the blood (mEq/L)
​
​
* **sex**: Sex of the patient
​
​
* **smoking**: If the patient smokes or not (Boolean)
​
​
* **time**: Indicates the number of days after a patient is admitted that he or she will die.
​
​
* **DEATH_EVENT**: If the patient deceased during the follow-up period (Boolean)
​
​
* **[Attributes having Boolean values: 0 = Negative (No); 1 = Positive (Yes)]**

# Project Report: Heart Failure Prediction

# 1. Introduction

**Background**

Heart failure is a serious medical condition that poses significant health risks to individuals. Early detection and prediction of heart failure are essential for timely medical interventions and improved patient outcomes. The project focuses on developing a predictive model using data science and machine learning techniques to accurately identify patients at risk of heart failure.

**Problem Statement**

The goal of this project is to build a predictive model that can identify individuals at risk of heart failure based on clinical and demographic features. This model can assist healthcare professionals in providing personalized care to patients and enhancing early detection of potential heart failure cases.

**Objective**

The primary objective of this project is to create a reliable and accurate predictive model for heart failure. Key steps involve data exploration, preprocessing, feature selection, model building, hyperparameter tuning, and performance evaluation.

**2. Data Exploration and Preprocessing**

**Dataset Overview**

The dataset contains 299 records and 13 columns, including features such as age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, time, and the target variable "DEATH_EVENT."

**Data Summary and Insights**
The dataset has been thoroughly explored and analyzed:

* Statistical summary of features
* Correlation analysis of variables
* Detection and handling of outliers
* Distribution analysis of continuous variables
* Assessment of multicollinearity

**3. Feature Selection**

Correlation analysis and feature importance techniques helped identify key features for model building. The optimal subset of features was selected to retain essential predictive power while avoiding overfitting.

**4. Data Splitting and Scaling**

The data was split into training and testing sets (80:20) to enable model evaluation. MinMax scaling was applied to ensure features were on the same scale, promoting convergence during model training.

**5. Handling Class Imbalance**

Due to class imbalance, the minority class was oversampled using the Synthetic Minority Over-sampling Technique (SMOTE) to enhance model performance.

**6. Model Building and Evaluation**

A variety of classification algorithms were evaluated, including Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, AdaBoost, Gradient Boosting, Random Forest, XGBoost, and Support Vector Classifier (SVC). Model performance was assessed using metrics such as accuracy, precision, recall, F1-score, and balanced accuracy.

**7. Hyperparameter Tuning**

Hyperparameter tuning was performed on the SVC model to optimize its performance. GridSearchCV was used to find the best combination of hyperparameters.

**8. Model Selection and Validation**

The Support Vector Classifier (SVC) was selected as the final model due to its robust performance and ability to handle imbalanced datasets. The model's performance was validated on both training and test sets.

**9. Performance Analysis**

The model's performance was thoroughly analyzed, including metrics such as accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC curves. Cross-validation results provided insight into the model's consistency.

**10. Feature Insights**

The importance of individual features in the model was analyzed, highlighting the most significant variables that contribute to accurate predictions.

**11. Conclusion**

The project successfully developed a predictive model for heart failure detection. The chosen SVC model demonstrated robust performance, effectively balancing accuracy and recall. Insights gained from feature analysis further reinforced the importance of certain variables in predicting heart failure.

**12. Project Documentation and Deployment**

The final trained model was saved as a pickle file for easy access and deployment. Recommendations were provided for deploying the model in a real-world healthcare setting.

In conclusion, the Heart Failure Prediction project leverages data science and machine learning techniques to build a predictive model that enhances early detection and intervention for heart failure patients. The project demonstrates the power of data-driven decision-making in healthcare, showcasing the potential to save lives and improve patient outcomes.




