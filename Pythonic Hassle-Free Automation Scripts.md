# Pythonic Hassle-Free Automation Scripts

The benevolent Python emerges as a veritable savior, graciously bestowing the gift of automation upon beleaguered analysts. Behold, in a place where time is money and efficiency is paramount, Python automation unfurls its cape in ten distinguished scenarios:

## 1. Data extraction and preparation

```python
import pandas as pd

# Define file paths
input_file = 'your_dataset.csv'    # Replace with the path to your Mockaroo dataset
output_file = 'cleaned_dataset.csv'  # Output file path for the cleaned dataset

# Load the dataset
df = pd.read_csv(input_file)

# Clean and transform the data
df['Email'] = df['Email'].str.strip().str.lower()
df['Phone'] = df['Phone'].str.replace(r'\D', '', regex=True)
df['Address'] = df['Address'].str.strip()

# Save the cleaned dataset to a new CSV file
df.to_csv(output_file, index=False)

# Display a success message
print(f"Dataset cleaned and saved to {output_file}")
```

```python
import mysql.connector
import pandas as pd
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Database connection parameters
db_params = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "your_database",
}

# Connect to MySQL database and retrieve data
try:
    conn = mysql.connector.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM your_table")
    data = cursor.fetchall()

    # Load data into a pandas DataFrame
    df = pd.DataFrame(data, columns=["Column1", "Column2", ...])  # Replace with actual column names

    # Data processing and anomaly detection (replace with your own logic)
    # Example: Using Isolation Forest for anomaly detection
    clf = IsolationForest(contamination=0.05)
    df['anomaly'] = clf.fit_predict(df[['Column1', 'Column2']])

    # Check if anomalies were detected
    if -1 in df['anomaly'].values:
        # Send warning email
        sender_email = 'your_email@gmail.com'
        receiver_email = 'data_team@example.com'
        password = 'your_email_password'

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = 'Anomalies Detected'

        body = 'Anomalies detected in the data. Please review.'
        message.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

    else:
        # Save cleaned data as a new CSV file
        df.to_csv('cleaned_data.csv', index=False)

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
```

## 2. Report generation

To generate an Excel report in the specified format, you can modify the previous Python script as follows. I'll assume you have a dataframe `combined_df` containing the necessary data for this report.

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

# Create a new Excel workbook
workbook = Workbook()
excel_writer = pd.ExcelWriter("sales_tracking_report.xlsx", engine="openpyxl")
excel_writer.book = workbook

# Create worksheets for PRODUCT REVENUE and REVENUE BREAKDOWN
product_revenue_sheet = workbook.active
product_revenue_sheet.title = "PRODUCT REVENUE"

revenue_breakdown_sheet = workbook.create_sheet(title="REVENUE BREAKDOWN")

# Define the data for PRODUCT REVENUE
product_revenue_data = combined_df[["PRODUCT NAME", "COST PER ITEM", "MARKUP PERCENTAGE", "TOTAL SOLD",
                                    "TOTAL REVENUE", "SHIPPING CHARGE PER ITEM", "SHIPPING COST PER ITEM",
                                    "PROFIT PER ITEM", "RETURNS", "TOTAL INCOME"]]

# Add headers to the PRODUCT REVENUE sheet
headers = product_revenue_data.columns.tolist()
product_revenue_sheet.append(headers)

# Add data to the PRODUCT REVENUE sheet
for row in dataframe_to_rows(product_revenue_data, index=False, header=False):
    product_revenue_sheet.append(row)

# Add headers and percentage data to the REVENUE BREAKDOWN sheet
revenue_breakdown_headers = ["", "ITEM 1", "ITEM 2", "ITEM 3", "ITEM 4", "ITEM 5", "ITEM 6", "ITEM 7", "ITEM 8", "ALL"]
revenue_breakdown_sheet.append(revenue_breakdown_headers)

percentage_data = ["TOTAL REVENUE"] + ["#DIV/0!" for _ in range(9)]
revenue_breakdown_sheet.append(percentage_data)

# Apply styling (optional)
for row in revenue_breakdown_sheet.iter_rows(min_row=1, max_row=1):
    for cell in row:
        cell.font = Font(bold=True)

# Save the Excel file
excel_writer.save()
excel_writer.close()

print("Excel report generated successfully.")
```

In this script:

1. We create separate sheets for "PRODUCT REVENUE" and "REVENUE BREAKDOWN" using `openpyxl`.

2. We extract the relevant data from the `combined_df` dataframe and add it to the "PRODUCT REVENUE" sheet.

3. We add headers to both sheets and set the percentage values in the "REVENUE BREAKDOWN" sheet.

4. Optional: We apply bold styling to the headers.

5. Finally, we save the Excel file as "sales_tracking_report.xlsx."

This script should generate an Excel report in the specified format with the data from your MySQL database and Mockaroo dataset. Please replace the placeholders in the code with your actual data source and filenames.

## 3. Proofread

To automate proofreading for a Word file using the `GingerIt` library in Python, you can follow these steps:

1. Install the `gingerit` library if you haven't already. You can install it using pip:

```python
pip install gingerit-py3
```

2. Create a Python script to read a Word document, extract text, and perform proofreading using `GingerIt`. Here's an example script:

```python
from docx import Document
from gingerit.gingerit import GingerIt

# Load the Word document
def load_docx(filename):
    doc = Document(filename)
    text = [p.text for p in doc.paragraphs]
    return "\n".join(text)

# Perform proofreading
def proofread_document(doc_text):
    parser = GingerIt()
    result = parser.parse(doc_text)

    corrected_text = result['result']
    corrections = result['corrections']

    return corrected_text, corrections

# Replace text in the document with corrected text
def replace_text_in_document(filename, corrected_text):
    doc = Document(filename)
    for paragraph, corrected_paragraph in zip(doc.paragraphs, corrected_text.split('\n')):
        paragraph.clear()  # Clear the existing content
        paragraph.add_run(corrected_paragraph)  # Add the corrected content

    doc.save('corrected_document.docx')

# Main function
if __name__ == "__main__":
    input_filename = 'your_document.docx'  # Replace with your Word document file
    doc_text = load_docx(input_filename)

    corrected_text, corrections = proofread_document(doc_text)

    # Print corrections (optional)
    for original, corrected in corrections:
        print(f"Original: {original}, Corrected: {corrected}")

    # Save the corrected text to a new document
    replace_text_in_document(input_filename, corrected_text)
    print("Proofreading completed. Corrected document saved as 'corrected_document.docx'")
```

This script defines three functions:

- `load_docx`: Loads the content of a Word document.
- `proofread_document`: Uses `GingerIt` to perform proofreading and obtain corrections.
- `replace_text_in_document`: Replaces the text in the original document with the corrected text and saves it as 'corrected_document.docx'.

Replace `'your_document.docx'` with the path to your Word document. When you run this script, it will proofread the document, print the corrections (optional), and save the corrected document as 'corrected_document.docx'.

## 4. Data integration 

## 5. Customer segmentation 

## 6. Anomaly detection

## 7. Sentiment analysis

## 8. Inventory management

## 9. Email campaigns

## 10. Predictive analytics

## 11. Data visualization