# Exploratory Data Analysis (EDA) Using Python

* Project in progress *

#### Table of Contents

- [EDA: Customer finance dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-customer-finance-dataset)
- [EDA: Dairy dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-dairy-dataset)
- [EDA: Titanic dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-titanic-dataset)


------------------------------

## EDA: Finance dataset
https://www.kaggle.com/datasets/dgawlik/nyse
EDA is highly regarded as an essential and strategic step in the data analysis and machine learning processes. To understand the significance of EDA, data scientists thoroughly examine and dissect the data to gain valuable insights, while also detecting patterns, relationships, correlations, or anomalies. This process prepares the data for modeling by incorporating tasks like data cleaning, processing, visualization, and feature tuning. Exploratory data analysis uncovers the various characteristics, relationships, and distributions present in the dataset. The acquired knowledge of the data, uncovered through EDA, can guide feature tuning, extract insights on data behavior, and ultimately enhance model performance.

### EDA Process Flowchat <br>
![EDA Process Flowchart](https://github.com/wusinyee/MW-Portfolio/blob/45bf57281e2cd3c052eef0a41a2a291d84c1cfcc/EDAML.drawio.png)

Here is how the EDA will proceed:
1. Data Collection: Aquire the dataset
2. Data cleaning and processing: Fill in the missing value. In this case, fill in the null / NaN rows with reasonable values. Perform data validation and mapping. Format the column name for consistency. 
3. Assess whether the dataset is usable
4. Statistics Technique: Find the mean, median, count distribution. Use the describe method and run statistical test
5. Relationship Analysis: Use bivariate and multivariate analysis
6. Visualization: Histogram, bar chart, boxplot, heatmap
7. Determine if the dataset is suitable for feature engineering based on the information value and  feature importance from multiple algos
8. Feature Engineering

-------------------------------


Get a general or big picture of the dataset, run a basic check on the column names, data type, null counts, and distinct values.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# df = pd.read_excel('data.xlsx')

def column_summary(df):
    summary_data = []
    
    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()
        
        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Example usage:
# Assuming df is your DataFrame
summary_df = column_summary(df)
display(summary_df)
```
[Image of summary_df]

Get addition value form the dataset.

```python
# Gets additional value such as min / median / max etc.
def column_summary_plus(df):
    result_df = pd.DataFrame(columns=['col_name', 'col_dtype', 'num_distinct_values',
                                      'min_value', 'max_value',
                                      'median_no_na', 'average_no_na','average_non_zero',
                                      'null_present', 'nulls_num', 'non_nulls_num',
                                      'distinct_values'])
    
    # Loop through each column in the DataFrame
    for column in df.columns:
        print(f"Start processing {column} col with {df[column].dtype} dtype")
        # Get column dtype
        col_dtype = df[column].dtype
        # Get distinct values and their counts
        value_counts = df[column].value_counts()
        distinct_values = value_counts.index.tolist()
        # Get number of distinct values
        num_distinct_values = len(distinct_values)
        # Get min and max values
        sorted_values = sorted(distinct_values)
        min_value = sorted_values[0] if sorted_values else None
        max_value = sorted_values[-1] if sorted_values else None

        # Get median value
        non_distinct_val_list = sorted(df[column].dropna().tolist())
        len_non_d_list = len(non_distinct_val_list)
        if len(non_distinct_val_list) == 0:
            median = None
        else:
            median = non_distinct_val_list[len_non_d_list//2]

        # Get average value if value is number
        if np.issubdtype(df[column].dtype, np.number):
            if len(non_distinct_val_list) > 0:
                average = sum(non_distinct_val_list)/len_non_d_list
                non_zero_val_list = [v for v in non_distinct_val_list if v > 0]
                average_non_zero = sum(non_zero_val_list)/len_non_d_list
            else:
                average = None
                average_non_zero = None
        else:
            average = None
            average_non_zero = None

        # Check if null values are present
        null_present = 1 if df[column].isnull().any() else 0

        # Get number of nulls and non-nulls
        num_nulls = df[column].isnull().sum()
        num_non_nulls = df[column].notnull().sum()

        # Distinct_values only take top 10 distinct values count
        top_10_d_v = value_counts.head(10).index.tolist()
        top_10_c = value_counts.head(10).tolist()
        top_10_d_v_dict = dict(zip(top_10_d_v,top_10_c))

        # Append the information to the result DataFrame
        result_df = result_df.append({'col_name': column, 'col_dtype': col_dtype, 'num_distinct_values': num_distinct_values, 
                                      'min_value': min_value, 'max_value': max_value,
                                      'median_no_na': median, 'average_no_na': average, 'average_non_zero': average_non_zero,
                                      'null_present': null_present, 'nulls_num': num_nulls, 'non_nulls_num': num_non_nulls,
                                      'distinct_values': top_10_d_v_dict}, ignore_index=True)
        
    return result_df

# Example usage:
# Assuming df is your DataFrame
summary_df = column_summary(df)
display(summary_df)
```

If there are any errors, it is most likely due to datatype of the pandas dataframe. When saving a pandas dataframe into a csvfile, and then loading it back again, make sure to save the datatype and reload it with the datatype.

```python
### To Save Pandas to CSV
def dtype_to_json(pdf, json_file_path: str) -> dict:
    '''
    Parameters
    ----------
    pdf : pandas.DataFrame
        pandas.DataFrame so we can extract the dtype
    json_file_path : str
        the json file path location
        
    Returns
    -------
    Dict
        The dtype dictionary used
    
    To create a json file which stores the pandas dtype dictionary for
    use when converting back from csv to pandas.DataFrame.
    '''
    dtype_dict = pdf.dtypes.apply(lambda x: str(x)).to_dict()
    
    with open(json_file_path, 'w') as json_file:
        json.dump(dtype_dict, json_file)
    
    return dtype_dict

def download_csv_json(df, mainpath):
    csvpath = f"{mainpath}.csv"
    jsonfp = f"{mainpath}_dtype.json"
    
    dtypedict = dtype_to_json(df, jsonfp)
    df.to_csv(csvpath, index=False)
    
    return csvpath, jsonfp


# Example usage:
download_csv_json(df, "/home/some_dir/file_1") 

### To Load CSV to Pandas
def json_to_dtype(jsonfilepath):
    with open(jsonfilepath, 'r') as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def csv_to_pandas(csvpath, jsonpath):
    dtypedict = json_to_dtype(jsonpath)
    pdf = pd.read_csv(csvpath,dtype=dtypedict)
    
    return pdf

# Example usage:
csvfp = "/home/some_dir/file_1.csv"
jsonfp = "/home/some_dir/file_1_dtype.json"
df = csv_to_pandas(csvfp, jsonfp)
```

One of the obvious issues is that the C_ID column was not a primary key, since the number of distinct values is not equal to the number of non-nulls. 
I am going to:
1) Get a general idea of what the dataset is like
2) Find the median / mean / rough statistical distribution
3) Check that there are no duplicated rows

```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(df.head())
print(df.describe())
print(df.duplicated().sum())
```
Just a quick Check.

```python
# Identify numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Perform univariate analysis on numerical columns
for column in numerical_columns:
    # For continuous variables
    if len(df[column].unique()) > 10:  # Assuming if unique values > 10, consider it continuous
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    else:  # For discrete or ordinal variables
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(x=column, data=df)
        plt.title(f'Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        
        # Annotate each bar with its count
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        textcoords = 'offset points')
        plt.show()
```

[Images of visualised dataset]

Dataset transformation

```python
### Rename the column names for familiarity
# This is if there is no requirement to use back the same column names.
# This is also only done if there is no pre-existing format, or if the col names don't follow conventional format.
# Normally will follow feature mart / dept format to name columns for easy understanding across board.

df_l1 = df.copy()
df_l1.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
new_col_dict = {'pc': 'c_pc', 'incm_typ': 'c_incm_typ', 'gn_occ': 'c_occ',
                 'num_prd': 'prod_nos', 'casatd_cnt': 'casa_td_nos', 'mthcasa': 'casa_bal_avg_mth',
                 'maxcasa': 'casa_bal_max_yr', 'mincasa': 'casa_bal_min_yr', 'drvcr': 'dr_cr_ratio_yr',
                 'mthtd': 'td_bal_avg', 'maxtd': 'td_bal_max', 'asset_value': 'asset_tot_val',
                 'hl_tag': 'loan_home_tag', 'al_tag': 'loan_auto_tag', 'pur_price_avg': 'prop_pur_price',
                 'ut_ave': 'ut_avg', 'maxut': 'ut_max', 'n_funds': 'funds_nos',
                 'cc_ave': 'cc_out_bal_avg_mth', 'max_mth_trn_amt': 'cc_txn_amt_max_mth', 'min_mth_trn_amt': 'cc_txn_amt_min_mth',
                 'avg_trn_amt': 'cc_txn_amt_avg_mth', 'ann_trn_amt': 'cc_txn_amt_yr', 'ann_n_trx': 'cc_txn_nos_yr'}
df_l1.rename(columns=new_col_dict, inplace=True)
```

Filling up missing or null values

```python
sns.set(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.boxplot(x='c_incm_typ', y='casa_bal_max_yr', data=df_l1)

# Set labels and title
plt.xlabel('Income Type')
plt.ylabel('casa_bal_max_yr')
plt.title('Boxplot of casa_bal_max_yr by Income Type')
plt.yscale('log')

# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
```
[Image of boxplot]

```python
new_df = df_l1[['prop_pur_price','loan_home_tag']]
null_loan_home = new_df[new_df['loan_home_tag'].isnull()]
not_null_count = null_loan_home[~null_loan_home[['prop_pur_price']].isnull().any(axis=1)].shape[0]
print("Number of rows where 'loan_home_tag' is null, but 'prop_pur_price' is not null:", not_null_count)

new_df = df_l1[['prop_pur_price','loan_home_tag']]
null_loan_home = new_df[new_df['prop_pur_price'].isnull()]
not_null_count = null_loan_home[~null_loan_home[['loan_home_tag']].isnull().any(axis=1)].shape[0]
print("Number of rows where 'prop_pur_price' is null, but 'loan_home_tag' is not null:", not_null_count)

new_df = df_l1[['prop_pur_price','loan_home_tag']]
condition = new_df['loan_home_tag'] == 1
new_df[condition].describe()
```

Changing DT

```python
dtype_mapping = {'c_id': str, 'c_age': int, 'c_pc': int, 'c_incm_typ': int, 'prod_nos': int,
                 'casa_td_nos': int, 'loan_home_tag': int, 'loan_auto_tag': int,
                 'funds_nos': int, 'cc_txn_nos_yr': int, 'u_id': int}
```

Validate the data TBC

Data Mapping for Categorial Features

```python
df_l1 = df.copy()
df_l1.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)
new_col_dict = {'pc': 'c_pc', 'incm_typ': 'c_incm_typ', 'gn_occ': 'c_occ',
                 'num_prd': 'prod_nos', 'casatd_cnt': 'casa_td_nos', 'mthcasa': 'casa_bal_avg_mth',
                 'maxcasa': 'casa_bal_max_yr', 'mincasa': 'casa_bal_min_yr', 'drvcr': 'dr_cr_ratio_yr',
                 'mthtd': 'td_bal_avg', 'maxtd': 'td_bal_max', 'asset_value': 'asset_tot_val',
                 'hl_tag': 'loan_home_tag', 'al_tag': 'loan_auto_tag', 'pur_price_avg': 'prop_pur_price',
                 'ut_ave': 'ut_avg', 'maxut': 'ut_max', 'n_funds': 'funds_nos',
                 'cc_ave': 'cc_out_bal_avg_mth', 'max_mth_trn_amt': 'cc_txn_amt_max_mth', 'min_mth_trn_amt': 'cc_txn_amt_min_mth',
                 'avg_trn_amt': 'cc_txn_amt_avg_mth', 'ann_trn_amt': 'cc_txn_amt_yr', 'ann_n_trx': 'cc_txn_nos_yr'}
df_l1.rename(columns=new_col_dict, inplace=True)
fill_values = {'c_edu': 'Unknown', 'c_hse': 'UNKNOWN', 'c_pc': 0, 'c_incm_typ': 0,
               'c_occ': 'UNKNOWN',
               'casa_td_nos': 0, 'casa_bal_avg_mth': 0, 'casa_bal_max_yr': 0, 'casa_bal_min_yr': 0,
               'td_bal_avg': 0, 'td_bal_max': 0,
               'loan_home_tag':0, 'loan_auto_tag': 0,
               'ut_avg': 0, 'ut_max': 0, 'funds_nos': 0,
               'cc_txn_amt_max_mth': 0, 'cc_txn_amt_min_mth': 0, 'cc_txn_amt_avg_mth': 0,
               'cc_txn_amt_yr': 0, 'cc_txn_nos_yr': 0, 'cc_lmt': 0}
df_l1.fillna(fill_values, inplace=True)
```

-------------------------------

## EDA: Dairy dataset
EDA allows us to gain valuable insights into dairy products, such as their qualities, nutritional content, and customer feedback. By analyzing a dataset containing information about various dairy products, we can obtain perspective from the data set's big picture, then understand their composition, pricing, and consumer preferences.


### EDA Process Flowchat <br>
![EDAprocess drawio](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/85608d10-95b3-4580-be3d-137953dc8b78)

The outcomes of this EDA have major implications for dairy industry stakeholders such as producers, marketers, and consumers. In a competitive market, the insights gained can help with informed decision-making and efficiency optimization.

## A systematic guide on performing EDA  <br>

1.	Importing Libraries:
* The code begins by importing the necessary libraries: pandas, numpy, matplotlib.pyplot, and seaborn. These libraries are commonly used for data analysis and visualization in Python.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

2.	Loading the Dataset:
* The dataset is loaded into a DataFrame using the pd.read_csv() function.

```python
df = pd.read_csv('dairy_products_dataset.csv')
```

3.	Understanding the Dataset:
*	The code snippet provides an overview of the dataset
*	It prints the number of rows and columns in the dataset using the shape attribute of the DataFrame
*	It displays the data types of each column using the dtypes attribute of the DataFrame  <br>
![step3numclmdatatypes](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/3733ad66-fd1e-4940-9c83-357488634b07) <br>

*	It presents the summary statistics of the numerical columns using the describe() method of the DataFrame   <br>
![step3sumstat](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/7987de03-b6fa-47d0-9517-e79b04c989fe)

```python
print("Number of rows and columns:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())
```
This revealed various summary statistics for the dataset:   <br>
The dataset includes both numerical and categorical variables, such as product type, fat content, price, weight, calories, protein content, calcium content, sodium content, carbohydrates, sugar content, cholesterol content, and customer ratings. These variables provide a comprehensive view of the dairy products and help us uncover patterns and distributions.  <br>

* The average price of dairy products is *$3.79*, with a standard deviation of **$1.68**
* The average weight is *590 grams*, with a standard deviation of **391 grams**
* In terms of nutritional content, the average calorie count is *370*, with a standard deviation of **319.37**
* The average protein content is *7.8 grams*, with a standard deviation of **7.16**
* The average calcium content is *232 mg*, with a standard deviation of *182.67*, and the average sodium content is *110 mg*, with a standard deviation of **56.57**
* The dataset also includes information on carbohydrates, sugar, and cholesterol, with their respective average values and standard deviations
* The average rating of the dairy products is *4.24*, with a standard deviation of **0.36**

4.	Handling Missing Data:
*	The code checks for missing values in the dataset:
*	It uses the isnull().sum() method to calculate the sum of missing values in each column of the DataFrame.
*	It prints the count of missing values for each column  <br>
![step4missing](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/5a735b48-10e4-411e-93a6-544440c57574)

```python
print("\nMissing values:\n", df.isnull().sum())
```
* Upon inspection, it was determined that the dataset does not contain any missing values. All columns are complete and have no null entries.

5.	Exploratory Data Analysis (EDA):
*	The code performs exploratory data analysis by creating example visualizations:
*	It creates a histogram of the 'Price' column using sns.histplot() and plt.show(), visualizing the distribution of prices   <br>
![step5distriprice](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/191bdd42-067d-407f-8823-3c550b3b0536)
*  A histogram of the 'Price' variable was plotted, illustrating the distribution of prices for the dairy products   <br>
![step5box](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/4a41aa70-f7d1-4f09-86c8-e37ce942b0bd)


```python
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'])
plt.title('Distribution of Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Rating', data=df)
plt.title('Rating by Dairy Product Type')
plt.show()
```
*	It creates a boxplot to examine the relationship between the 'Rating' and 'Type' columns using sns.boxplot() and plt.show()
* A box plot was generated to explore the relationship between the 'Type' of dairy products (Cheese, Milk, Butter, Yogurt, Ice Cream) and their corresponding ratings

6.	Correlation Analysis:
*	The code calculates the correlation matrix of the dataset using df.corr().
*	It creates a heatmap visualization of the correlation matrix using sns.heatmap() and plt.show(). The heatmap provides a visual representation of the correlations between different numerical variables in the dataset   <br>

![step7corrmatrix](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/bbafb859-791b-4cf8-893b-19e51b85ebd0)


```python
correlation_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

7.	Feature Engineering (Optional):
*	The code demonstrates an example of feature engineering by creating a new feature called 'Price per Weight'. It calculates the price per weight by dividing the 'Price' column by the 'Weight' column.

```python
df['Price per Weight'] = df['Price'] / df['Weight']
```

8.	Additional Visualizations:
*	The code creates additional visualizations to explore relationships between variables:
*	It creates a scatter plot of 'Calories' versus 'Protein' with different colors representing different 'Type' categories using sns.scatterplot() and plt.show()   <br>
![step8](https://github.com/wusinyee/SYW-Portfolio-v2023/assets/108232087/47bd6ae0-b3de-46d9-b892-feb4b57eb12b)

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Calories', y='Protein', data=df, hue='Type')
plt.title('Calories vs Protein by Dairy Product Type')
plt.show()
```
* A scatter plot was utilized to visualize the relationship between 'Calories' and 'Protein' across different types of dairy products

### Relationship and Correlation:   <br>
The box plot analysis indicated that the various types of dairy products exhibit different ratings. Specifically, Cheese demonstrated the highest median rating, while Butter had the lowest. The scatter plot analysis revealed a positive relationship between the 'Calories' and 'Protein' content of dairy products. Generally, as the calorie content increases, so does the protein content. 
*Further exploration of potential relationships and correlations can be conducted by examining additional variables present in the dataset.*

## Key Insights and Recommendations 
1.	Marketing Tactics: The EDA results provide information about the distribution of prices for dairy products, allowing marketers to understand the price range and make informed decisions about pricing strategies. Additionally, the box plot analysis of ratings by product type can help identify which types of dairy products have higher ratings, enabling marketers to focus their marketing efforts on promoting those products. Understanding the relationship between calories and protein content can also guide marketing tactics by highlighting the nutritional benefits of certain products.   <br>
2.	Best and Worst Product: By examining the ratings of different dairy products, as shown in the box plot, it is possible to identify the best and worst products. In this case, Cheese has the highest median rating, indicating it is likely the best product, while Butter has the lowest median rating, suggesting it may be the worst product. Marketers can leverage this information to highlight the positive aspects of the best product and strategize ways to improve or reposition the worst product.   <br>
3.	Most and Least Profitable Products: The EDA does not provide direct information about the profitability of products, as it focuses on price, nutritional content, and ratings. However, marketers can correlate the pricing information with sales data or profit margins to estimate the profitability of each product. By analyzing the relationship between price and other variables, marketers can identify potential opportunities for increasing profitability, such as adjusting prices or promoting higher-priced products with desirable characteristics.    <br>
4.	Healthiest Product: Determining the healthiest product can be subjective and depends on specific health criteria. However, the EDA provides insights into the nutritional content of dairy products, such as calories, protein, carbohydrates, sugar, and cholesterol. Marketers can use this information to position certain products as healthier options based on their nutritional profiles. For example, a product with lower levels of sugar and cholesterol or higher protein content may be promoted as a healthier choice.    <br>
5.	Pain Points: Identifying pain points requires additional information beyond what is provided by the EDA. Pain points may include customer complaints, issues with product quality, or gaps in the market. The EDA can serve as a starting point for identifying potential pain points by analyzing the ratings and understanding customer preferences. By identifying products with lower ratings or observing patterns in customer feedback, marketers can gain insights into potential pain points and areas for improvement.


-------------------------------

## EDA: Titanic dataset

Reference
https://medium.com/towardsdev/python-data-visualization-for-exploratory-data-analysis-eda-a25a94d73687




-------------------------------




