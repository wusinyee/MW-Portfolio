# Exploratory Data Analysis (EDA)

- Analyze the distribution of numerical features (tenure, MonthlyCharges, TotalCharges)
- Examine the relationship between these features and the target variable Churn
- Create visualizations to show the impact of categorical variables on churn rate

### Dataset Summary
- **Dataset Name:** WAFn-UseC-Telco-Customer-Churn.csv
- **Number of Rows:** 7043
- **Number of Columns:** 21

### Column Descriptions
1. **customerID:** Unique identifier for each customer.
2. **gender:** Gender of the customer (Male, Female).
3. **SeniorCitizen:** Indicates if the customer is a senior citizen (1) or not (0).
4. **Partner:** Whether the customer has a partner (Yes, No).
5. **Dependents:** Whether the customer has dependents (Yes, No).
6. **tenure:** Number of months the customer has stayed with the company.
7. **PhoneService:** Whether the customer has a phone service (Yes, No).
8. **MultipleLines:** Whether the customer has multiple lines (No phone service, No, Yes).
9. **InternetService:** Customer’s internet service provider (DSL, Fiber optic, No).
10. **OnlineSecurity:** Whether the customer has online security service (Yes, No, No internet service).
11. **OnlineBackup:** Whether the customer has online backup service (Yes, No, No internet service).
12. **DeviceProtection:** Whether the customer has device protection (Yes, No, No internet service).
13. **TechSupport:** Whether the customer has tech support (Yes, No, No internet service).
14. **StreamingTV:** Whether the customer has streaming TV service (Yes, No, No internet service).
15. **StreamingMovies:** Whether the customer has streaming movies service (Yes, No, No internet service).
16. **Contract:** Type of contract the customer has (Month-to-month, One year, Two year).
17. **PaperlessBilling:** Whether the customer uses paperless billing (Yes, No).
18. **PaymentMethod:** Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
19. **MonthlyCharges:** The amount charged to the customer monthly.
20. **TotalCharges:** The total amount charged to the customer.
21. **Churn:** Whether the customer has churned (Yes, No).

## Initial Observations
- **Data Types:** Most columns are of object type except for SeniorCitizen (int64), tenure (int64), MonthlyCharges (float64), and TotalCharges (object, which should be numeric).
- **Missing Values:** No missing values in any columns initially, but after converting TotalCharges to numeric, there are 11 missing values.
- **Target Variable:** The target variable for churn prediction is Churn.

## Conversion and Cleaning
- **TotalCharges Conversion:** Converting TotalCharges to numeric resulted in 11 missing values. These rows can be inspected and handled by imputing or dropping them.

## Statistical Summary
- **Numerical Features:** Summary statistics for numerical columns (tenure, MonthlyCharges, TotalCharges).
- **Categorical Features:** Unique values and distributions of categorical columns (gender, SeniorCitizen, Partner, etc.).

----------------------------------

1. **Import necessary libraries and load the data**

```python
!pip install plotly ipywidgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interactive
from IPython.display import display, HTML
# Load the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
```


2. **Initial data inspection**

```python
# Display basic information about the dataset
print(df.info())

# Display the data description
print(df.describe())

# Display the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())
```



