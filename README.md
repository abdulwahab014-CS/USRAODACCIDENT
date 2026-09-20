# Railroad Highway-Crossing Incident Data Pipeline

A comprehensive Python-based data cleaning, preprocessing, and feature engineering pipeline built for analyzing railroad highway-crossing collision records. This repository converts raw, noisy, and unformatted incident data into an analysis-ready, standardized dataset optimized for machine learning models (e.g., predicting accident severity, injury outcomes, or property damage costs).

---

## Project Overview

Raw highway-rail collision records maintained by transportation safety authorities often contain data quality issues like missing records, unformatted dates, duplicate entries, extreme financial/speed outliers, and unencoded categorical features. 

This project implements a modular **42-cell automated preprocessing pipeline** in Google Colab that systematically cleans the dataset, engineer 8+ domain-specific safety features, handles outliers via IQR clipping, and standardizes continuous attributes for predictive modeling.

---

##  Objectives

* **Data Quality Enhancement:** Eliminate missing values, duplicate entries, and inconsistent string formatting.
* **Feature Engineering:** Derive meaningful Domain-Specific metrics such as accident severity scores, speed differentials, time-of-day indicators, and per-occupant damage metrics.
* **Outlier Mitigation:** Identify extreme speed and financial outliers using boxplots and bound them using Interquartile Range (IQR) clipping.
* **Encoding & Scaling:** Encode categorical text attributes into numeric formats and scale numerical variables ($\mu = 0, \sigma = 1$) using `StandardScaler`.
* **Automated Pipeline Creation:** Construct a reproducible Scikit-Learn `Pipeline` and `ColumnTransformer` workflow.

---

##  Technologies Used

* **Language:** Python 3.x
* **Environment:** Google Colab / Jupyter Notebook
* **Data Manipulation & Analysis:** `Pandas`, `NumPy`
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning & Preprocessing:** `Scikit-Learn`
  * `StandardScaler`
  * `LabelEncoder`, `OneHotEncoder`
  * `SimpleImputer`, `Pipeline`, `ColumnTransformer`

---

##  Dataset Description

The dataset contains historical records of highway-rail crossing collisions involving driver demographics, environmental conditions, train specifications, and impact outcomes:

* **Temporal & Spatial:** `date`, `time`, `incident_datetime`, `state_name`
* **Driver & Highway User:** `user_age`, `user_gender`, `highway_user`, `estimated_vehicle_speed`, `number_vehicle_occupants`
* **Train & Railway Properties:** `train_speed`, `railroad_type`, `equipment_type`, `number_of_cars`, `number_of_locomotive_units`
* **Environmental Factors:** `weather_condition`, `temperature`
* **Impact & Damage Metrics:** `vehicle_damage_cost`, casualty counts (injured/killed across users, employees, passengers)

---

##  Data Preprocessing Pipeline Steps

1. **Data Loading & Deduplication:** Removes duplicate rows (`df.drop_duplicates()`) and standardizes column headers into `snake_case`.
2. **Missing Value Imputation:** Substitutes missing numerical attributes using median values and fills missing categorical text with `"Unknown"`.
3. **Data Type Correction:** Converts string date columns into native Pandas `datetime64` objects and forces numerical fields into float/integer types.
4. **Outlier Capping:** Detects extreme values in speeds and damage costs, applying IQR clipping ($Q1 - 1.5 \times \text{IQR}$ to $Q3 + 1.5 \times \text{IQR}$).
5. **Feature Engineering (8+ New Features):**
   * `incident_datetime`: Combined `date` and `time` into a unified timestamp.
   * `is_night`: Binary flag indicating nighttime collisions (hours $< 6$ or $\ge 18$).
   * `is_weekend`: Binary flag identifying weekend incidents.
   * `speed_difference`: `estimated_vehicle_speed` $-$ `train_speed`.
   * `total_injuries`: Combined injuries across users, employees, and passengers.
   * `total_fatalities`: Combined fatalities across users, employees, and passengers.
   * `damage_per_occupant`: `vehicle_damage_cost` divided by total occupants.
   * `accident_severity`: Weighted metric ($\text{total\_injuries} + 3 \times \text{total\_fatalities}$).
6. **Encoding & Scaling:** Applies `LabelEncoder` to categorical attributes and standardizes continuous numerical attributes via `StandardScaler`.
7. **Automated Scikit-Learn Pipeline:** Implements a reusable `ColumnTransformer` for production integration.

---

##  Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
