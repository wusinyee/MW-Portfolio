# Exploratory Data Analysis (EDA) Using Python

#### Table of Contents

- [EDA: Customer finance dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-customer-finance-dataset)
- [EDA: Dairy dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-dairy-dataset)
- [EDA: Titanic dataset](https://github.com/wusinyee/MW-Portfolio/edit/main/EDAwithPython.md#eda-titanic-dataset)


------------------------------

## EDA: Finance dataset
EDA is highly regarded as an essential and strategic step in the data analysis and machine learning processes. To understand the significance of EDA, data scientists thoroughly examine and dissect the data to gain valuable insights, while also detecting patterns, relationships, correlations, or anomalies. This process prepares the data for modeling by incorporating tasks like data cleaning, processing, visualization, and feature tuning. Exploratory data analysis uncovers the various characteristics, relationships, and distributions present in the dataset. The acquired knowledge of the data, uncovered through EDA, can guide feature tuning, extract insights on data behavior, and ultimately enhance model performance.

### EDA Process Flowchat <br>
![EDA Process Flowchart](https://github.com/wusinyee/MW-Portfolio/blob/45bf57281e2cd3c052eef0a41a2a291d84c1cfcc/EDAML.drawio.png)


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







-------------------------------




