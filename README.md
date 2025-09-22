# Supervised Learning Model for Application Fraud Detection

![Project Status](https://img.shields.io/badge/status-complete-success)

This repository contains the complete workflow for building and evaluating a supervised machine learning model to detect fraudulent credit card and cell phone applications. The project progresses from initial data quality assessment to advanced feature engineering, comparative model analysis, and culminates in a financial impact analysis to determine an optimal operational strategy.

The final XGBoost model successfully identifies **60% of fraudulent applications** in an out-of-time (OOT) test set by reviewing only the **top 3% of riskiest applications**, translating to an estimated **$3 billion in annual fraud savings**.

---

### **Table of Contents**
1.  [Project Overview](#project-overview)
2.  [Key Results](#key-results)
3.  [Repository Structure](#repository-structure)
4.  [Methodology & Technical Details](#methodology--technical-details)
    - [Data Quality Assessment](#1-data-quality-assessment)
    - [Data Cleaning & Preprocessing](#2-data-cleaning--preprocessing)
    - [Feature Engineering](#3-feature-engineering)
    - [Feature Selection](#4-feature-selection)
    - [Model Exploration & Selection](#5-model-exploration--selection)
    - [Final Model & Performance](#6-final-model--performance)
5.  [Financial Impact Analysis](#financial-impact-analysis)
6.  [Professional Reports](#professional-reports)
7.  [Key Skills Demonstrated](#key-skills-demonstrated)
8.  [How to Run](#how-to-run)

---

### **Project Overview**

The goal of this project was to address the significant business problem of application fraud, where individuals use stolen or synthetic identities to open credit card or cell phone accounts. By leveraging a year of historical application data, we developed a robust machine learning model capable of flagging high-risk applications before they result in financial losses. The project emphasizes not just predictive accuracy but also the practical application of the model within a business context, balancing fraud detection rates with operational costs.

---

### **Key Results**

* **High Detection Rate:** The final XGBoost model captures approximately **60% of all fraud cases** on a completely unseen, out-of-time dataset by flagging only the top 3% of applications for manual review.
* **Substantial Financial Impact:** A financial analysis determined an optimal review cutoff at the top 1% of riskiest applications. This strategy is estimated to prevent **$3 billion in annual fraud losses** after accounting for investigation costs.
* **Model Robustness:** The model demonstrated strong generalization capabilities with minimal performance degradation between the training, testing, and out-of-time (OOT) datasets, indicating its stability over time.

---

* **notebooks/:** Contains Jupyter notebooks that cover the entire project pipeline, from data analysis to final model evaluation.
* **reports/:** Includes detailed PDF reports documenting the project's findings, data quality, and methodology.
* **data/:** A directory for the raw, intermediate, and final datasets. *(Note: The dataset is synthetic and not included in this repository to simulate a real-world scenario where data is confidential.)*

---

### **Methodology & Technical Details**

#### **1. Data Quality Assessment**
The project began with a thorough analysis of the raw dataset (1,000,000 application records, 1.5% fraud rate) as detailed in the `Data_Quality_Report.pdf`. This involved examining field completeness, identifying placeholder values (e.g., `999999999` for SSN), and analyzing distributions to ensure data integrity before modeling.

#### **2. Data Cleaning & Preprocessing**
Key cleaning steps were performed to prepare the data for feature engineering:
* **Standardization:** Text fields like names and addresses were lowercased and trimmed to ensure consistency.
* **Handling Frivolous PII:** Placeholder values in key identifier fields (SSN, Phone, Address) were replaced with nulls. This is critical to prevent false linkages between unrelated fraudulent applications that use common dummy values.
* **Composite Key Creation:** New entity keys were constructed by concatenating fields (e.g., `fullname`, `name_dob`, `address_zip`) to create more precise identifiers for linking applications.

#### **3. Feature Engineering**
Over 170 features were engineered to capture complex fraud behaviors, primarily focusing on the velocity and recency of information usage. This is the most critical step for model performance.

* **Velocity Features:** These count the number of times an entity (like an SSN or phone number) has appeared in recent time windows (0, 1, 3, 7, 14, and 30 days). High velocity in a short period is a strong indicator of fraudulent activity.
    * *Example:* `ssn_count_7` = Number of applications in the past 7 days with the same SSN.

* **Recency Features (Days Since Last Occurrence):** This measures the time elapsed since an entity was last seen. A very small value (e.g., 0 or 1 day) indicates rapid, suspicious reuse of an identity.
    * *Example:* `fulladdress_days_since`.

* **Relative Velocity Ratios:** These are ratios of counts across different time windows (e.g., `count_1_day / count_7_day`). They capture the *acceleration* or deceleration of activity, highlighting sudden spikes.

* **Cross-Entity Unique Counts:** These powerful features measure the diversity of relationships between identifiers, which is effective at detecting fraud rings.
    * *Example:* `ssn_unique_address_count` = For a given SSN, how many unique addresses has it been associated with historically? A high count is highly suspicious.

* **Day-of-Week Fraud Risk:** To capture temporal patterns without overfitting, a smoothed fraud rate for each day of the week was calculated using a logistic smoothing function:
    $$
    y_{\text{dow\_smooth}} = \bar{y} + \frac{y_{\text{dow}} - \bar{y}}{1 + \exp(-\frac{\text{num} - n_{\text{mid}}}{c})}
    $$
    This provides a robust risk score for the application day while mitigating noise from low-volume days.

#### **4. Feature Selection**
A two-stage feature selection process was employed to reduce the feature set from 176+ to a robust and computationally efficient set of 20 variables.
1.  **Filter Method:** An initial ranking was created using the univariate Kolmogorov-Smirnov (KS) statistic to measure each feature's ability to separate fraudulent from legitimate applications.
2.  **Wrapper Method:** A **Sequential Feature Selector (SFS)** was used with a LightGBM model to iteratively build the optimal feature set, accounting for interactions between variables that univariate methods miss.

#### **5. Model Exploration & Selection**
A wide range of classification algorithms were trained and evaluated to identify the best-performing model architecture.
* **Models Tested:** Logistic Regression (Baseline), Decision Tree, Random Forest, Multi-Layer Perceptron (MLP), XGBoost, LightGBM, and CatBoost.
* **Evaluation Metric:** The primary metric for comparison was the **Fraud Detection Rate at 3% Review (FDR@3%)**. This business-centric metric measures the percentage of total fraud dollars captured by reviewing the top 3% of applications flagged by the model, aligning directly with operational capacity constraints.
* **Winner:** **XGBoost** was selected as the final model due to its superior performance, speed, and robust generalization on the validation set.

#### **6. Final Model & Performance**
The final XGBoost model was tuned using 5-fold cross-validation. Key hyperparameters included:
* `max_depth`: 6
* `learning_rate`: 0.10
* `subsample`: 0.80
* `n_estimators`: 150 (with early stopping)
* `scale_pos_weight`: 65 (to handle class imbalance)

The model's final performance was validated on three distinct datasets:

| Metric  | Training Set | Independent Test Set | Out-of-Time (OOT) Set |
| :------ | :----------: | :------------------: | :-------------------: |
| **AUC** |     0.99     |         0.98         |         0.98          |
| **FDR@3%**|     65%      |          62%         |          60%          |

The stable performance on the OOT set (Nov-Dec 2017 data) confirms the model's ability to generalize to new, unseen data beyond the training period.

---

### **Financial Impact Analysis**
To translate the model's statistical performance into business value, a financial analysis was conducted. This involved plotting three curves against the percentage of applications reviewed:
1.  **Fraud Losses Prevented:** The total dollar amount of fraud stopped.
2.  **Investigation Cost:** The operational cost of manually reviewing applications.
3.  **Net Savings:** The difference between losses prevented and costs incurred.

The analysis revealed that **net savings are maximized at a 1% review rate**. At this optimal cutoff, the model captures **~54% of all fraud**, yielding an estimated **$3 billion in annual net savings**. This provides a clear, data-driven recommendation for model deployment in a business environment.

---

### **Professional Reports**
This repository includes professional reports that document the project in detail. These are intended to showcase the ability to communicate complex technical findings to both technical and business audiences.
* **[`Final_Report.pdf`](Final_Report.pdf):** A comprehensive summary of the entire project, including methodology, model performance, and financial analysis.
* **[`Data_Quality_Report.pdf`](DQR.pdf):** A detailed field-by-field analysis of the raw data.
* **[`Data_Cleaning_Variable_Creation.pdf`](Data_Cleaning_Variable_Creation.pdf):** A technical document outlining the preprocessing and feature engineering pipeline.

---

### **Key Skills Demonstrated**

* **Data Science & Machine Learning:**
    * Supervised Learning (Binary Classification)
    * Fraud Detection & Anomaly Detection
    * Advanced Feature Engineering (Velocity, Recency, Relational Features)
    * Feature Selection (Filter & Wrapper Methods)
    * Model Training, Tuning, and Evaluation (XGBoost, LightGBM, CatBoost, etc.)
    * Handling Class Imbalance (`scale_pos_weight`)
    * Time-Series Validation (Out-of-Time Testing)

* **Tools & Technologies:**
    * **Python:** Pandas, NumPy, Scikit-learn
    * **ML Frameworks:** XGBoost, LightGBM, CatBoost
    * **Visualization:** Matplotlib, Seaborn
    * **Development:** Jupyter Notebooks

* **Analytical & Business Skills:**
    * Data Quality Assessment
    * Business Impact Modeling & ROI Analysis
    * Selection of Business-Relevant Metrics (FDR@3%)
    * Technical Documentation & Reporting

---

### **How to Run**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Application-Fraud-Detection-Model.git](https://github.com/your-username/Application-Fraud-Detection-Model.git)
    cd Application-Fraud-Detection-Model
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created listing packages like pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn, etc.)*

4.  **Run the Jupyter Notebooks** in the `notebooks/` directory in sequential order.
