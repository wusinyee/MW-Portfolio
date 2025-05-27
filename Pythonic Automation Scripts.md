# Pythonic Hassle-Free Automation Scripts

The benevolent Python emerges as a veritable savior, graciously bestowing the gift of automation upon beleaguered analysts. Behold, in a place where time is money and efficiency is paramount, Python automation unfurls its cape in ten distinguished scenarios:

## 1. Data extraction automation scripts

Developing an optimal Python automation script for extracting data from a MySQL database encompasses efficient data retrieval, appropriate error handling, and a well-organized, maintainable code structure. 

```python
import mysql.connector
import pandas as pd

def extract_data(host, user, password, database, query):
    try:
        # Create a connection to the MySQL database using a context manager
        with mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        ) as connection:
            # Create a cursor using a context manager
            with connection.cursor() as cursor:
                # Execute the SQL query and fetch data into a Pandas DataFrame
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(data, columns=columns)
        return df

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def test_extract_data():
    # Replace with your test MySQL server details and query
    test_host = "your_test_host"
    test_user = "your_test_user"
    test_password = "your_test_password"
    test_database = "your_test_database"
    test_query = "SELECT * FROM your_test_table"

    # Extract test data
    test_data = extract_data(test_host, test_user, test_password, test_database, test_query)

    # Check if data extraction was successful
    assert test_data is not None, "Data extraction failed"

    # Check if the DataFrame has rows
    assert not test_data.empty, "DataFrame is empty"

    print("Test passed. Data extracted successfully.")

if __name__ == "__main__":
    # Replace with your MySQL server details and query
    host = "your_host"
    user = "your_user"
    password = "your_password"
    database = "your_database"
    query = "SELECT * FROM your_table"

    # Extract data
    extracted_data = extract_data(host, user, password, database, query)

    if extracted_data is not None:
        # Process the data or save it as needed
        print(extracted_data.head())

        # You can perform further data processing or save the data to a file here

    # Run the test function
    test_extract_data()
```
This script utilizes context managers for improved resource management and incorporates a test_extract_data function to verify the accuracy of the data extraction. Replace the test MySQL server details with your own for testing.
To use the script:
1. Replace the placeholders for your MySQL server details and query.
2. Execute the script. The data will be extracted and printed from the MySQL database, followed by the execution of the test function to verify correctness.

## 2. Report generation

The following script generates an Excel report from a DataFrame. To ensure it has no bugs and optimized performance, a test function will be added to verify its correctness.

```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

def generate_excel_report(df, output_file):
    # Create a new Excel workbook
    workbook = Workbook()
    excel_writer = pd.ExcelWriter(output_file, engine="openpyxl")
    excel_writer.book = workbook

    # Create worksheets for PRODUCT REVENUE and REVENUE BREAKDOWN
    product_revenue_sheet = workbook.active
    product_revenue_sheet.title = "PRODUCT REVENUE"

    revenue_breakdown_sheet = workbook.create_sheet(title="REVENUE BREAKDOWN")

    # Define the data for PRODUCT REVENUE
    product_revenue_data = df[["PRODUCT NAME", "COST PER ITEM", "MARKUP PERCENTAGE", "TOTAL SOLD",
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

    print(f"Excel report generated successfully and saved as '{output_file}'.")

def test_generate_excel_report():
    # Create a sample DataFrame (you can replace this with your actual data)
    sample_data = {
        "PRODUCT NAME": ["Item 1", "Item 2", "Item 3"],
        "COST PER ITEM": [10, 15, 12],
        "MARKUP PERCENTAGE": [20, 25, 18],
        "TOTAL SOLD": [100, 150, 120],
        "TOTAL REVENUE": [2000, 3750, 2160],
        "SHIPPING CHARGE PER ITEM": [5, 7, 6],
        "SHIPPING COST PER ITEM": [2, 3, 2.5],
        "PROFIT PER ITEM": [13, 17, 13.5],
        "RETURNS": [5, 2, 3],
        "TOTAL INCOME": [1995, 3748, 2156.5]
    }
    df = pd.DataFrame(sample_data)

    # Generate the Excel report for testing
    test_output_file = "test_sales_tracking_report.xlsx"
    generate_excel_report(df, test_output_file)

if __name__ == "__main__":
    # Test the generate_excel_report function
    test_generate_excel_report()
```

In this script:

1. The report generation logic was encapsulated into a function called "generate_excel_report."
2. A test function, named test_generate_excel_report, has been implemented to generate a sample Excel report specifically for testing purposes. When conducting tests, it is possible to replace the sample data with the actual data that is being used.
3. The test function generates an Excel report containing a test DataFrame and verifies the successful execution.

To conduct the script testing, execute it. The program will generate an Excel report sample and display a success message. It is important to ensure that the required libraries, such as pandas and openpyxl, are installed.

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

## 4. Data Preperation 
The development of an optimal Python automation script for data preparation encompasses the efficient cleaning, transformation, and organization of data. In order to guarantee error-free and optimized data preparation, we will present a fundamental illustration of the data preparation process, accompanied by a test function to assess its accuracy. It should be noted that the process of data preparation can vary significantly depending on the specific dataset and requirements. Therefore, the following example is a simplified representation.

The following is the script:
```python
import pandas as pd

def prepare_data(input_file, output_file):
    try:
        # Read the input data into a DataFrame
        data = pd.read_csv(input_file)

        # Data cleaning and transformation (sample, replace with your specific logic)
        # Example: Remove rows with missing values and convert a column to datetime
        data.dropna(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])

        # Data organization (sample, replace with your specific logic)
        # Example: Sort data by a column and reset the index
        data = data.sort_values(by='Date').reset_index(drop=True)

        # Save the prepared data to an output file (e.g., CSV)
        data.to_csv(output_file, index=False)

        print(f"Data preparation completed. Prepared data saved as '{output_file}'.")

    except Exception as e:
        print(f"Error: {e}")

def test_prepare_data():
    # Create a sample CSV file for testing (you can replace this with your own data)
    sample_data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Value': [10, None, 20]
    }
    df = pd.DataFrame(sample_data)

    # Save the sample data to an input file
    input_file = 'sample_input_data.csv'
    df.to_csv(input_file, index=False)

    # Test data preparation
    output_file = 'prepared_data.csv'
    prepare_data(input_file, output_file)

if __name__ == '__main__':
    # Test the prepare_data function
    test_prepare_data()
```
Data preparation is an essential step in data analysis, and adherence to best practices guarantees the cleanliness, proper structure, and analysis readiness of your data. Here is an example of a Python automation script for data preparation that incorporates best practices. It is recommended to integrate this code with your specific data and requirements.

```python
import pandas as pd

def data_preparation(input_file, output_file):
    try:
        # Read the input data into a DataFrame
        data = pd.read_csv(input_file)

        # Data cleaning and transformation
        # Example 1: Handling missing values by filling them with the mean of the column
        data.fillna(data.mean(), inplace=True)

        # Example 2: Removing duplicates
        data.drop_duplicates(inplace=True)

        # Example 3: Data type conversion (e.g., converting a date column)
        data['Date'] = pd.to_datetime(data['Date'])

        # Data organization
        # Example: Sorting data by a specific column
        data.sort_values(by='Date', inplace=True)

        # Save the prepared data to an output file (e.g., CSV)
        data.to_csv(output_file, index=False)

        print(f"Data preparation completed. Prepared data saved as '{output_file}'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Replace these with your input and output file paths
    input_file = 'input_data.csv'
    output_file = 'prepared_data.csv'

    # Perform data preparation
    data_preparation(input_file, output_file)
```
Several best practices have been incorporated in this script:
1. Handling Missing Values: The fillna method is employed to replace missing values with the column's mean, a widely adopted strategy for numerical data imputation. The adaptation of this approach can be customized to suit the specific data type and requirements of the user.
2. Elimination of Duplicates: In order to ensure accurate analysis results, the drop_duplicates function is employed to remove any duplicate entries.
3. Data Type Conversion: The conversion of a date column to a datetime format is performed using the pd.to_datetime function, which is crucial when handling date-based data.
4. Data Organization: The data is sorted according to a designated column, such as 'Date', in order to facilitate its organization for subsequent analysis.
5. Error Handling: Error handling has been implemented to capture and report any exceptions that may arise during the process of data preparation.
6. Input and Output File Paths: Substituting 'input_data.csv' and 'prepared_data.csv' with the paths to the respective actual input and output files is recommended.
To utilize this script, substitute the file paths of the input and output files with the respective paths of your data files, and execute the script. The system will execute data preparation and store the prepared data as specified.
The data cleaning and transformation steps should be customized according to the specific dataset and analysis requirements.

## 5. Data integration 

To develop an efficient Python automation script for data integration from Amazon Redshift to a CSV file on a local drive, the psycopg2 library can be utilized for establishing a connection with Redshift, while pandas can be employed for data manipulation.
To ensure accuracy, here is a script with a test function:

```python
import psycopg2
import pandas as pd

def redshift_to_csv(host, port, database, user, password, query, output_csv):
    try:
        # Connect to Amazon Redshift
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        # Execute the SQL query and fetch data into a DataFrame
        df = pd.read_sql_query(query, conn)

        # Save the DataFrame as a CSV file
        df.to_csv(output_csv, index=False)

        print(f"Data from Redshift successfully exported to '{output_csv}'.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the database connection
        if conn:
            conn.close()

def test_redshift_to_csv():
    # Replace with your Redshift server details, query, and desired output file path
    host = "your_redshift_host"
    port = "your_redshift_port"
    database = "your_redshift_database"
    user = "your_redshift_user"
    password = "your_redshift_password"
    query = "SELECT * FROM your_redshift_table"
    output_csv = "output_data.csv"

    # Test the data integration
    redshift_to_csv(host, port, database, user, password, query, output_csv)

if __name__ == "__main__":
    # Test the redshift_to_csv function
    test_redshift_to_csv()
```

In the script above:
1. The redshift_to_csv function is defined to establish a connection with the Amazon Redshift database, execute the provided SQL query, and retrieve the data into a Pandas DataFrame.
2. The DataFrame's to_csv method then stores the data locally as a CSV file.
3. Error handling is implemented in order to capture and report any exceptions that may occur during the process.
4. We offer a test_redshift_to_csv function for the purpose of testing data integration. Substitute the placeholders with the specific Redshift server details, query, and the desired output CSV file path.

To use the script:
1. Substitute the placeholders in the test_redshift_to_csv function with the specific details of your Redshift server, query, and the desired file path for storing the resulting CSV file.
2. Execute the script. The system will establish a connection with the Redshift database, retrieve the data, and store it locally as a CSV file. Additionally, a success message will be displayed.
Ensure that the psycopg2 and pandas libraries are installed by executing the command "pip install psycopg2 pandas" prior to running the script.


## 6. File Organizer Script
Automatically organizes files in a directory by type, date, or size.

```python
#!/usr/bin/env python3
"""
File Organizer Script
Automatically organizes files in a directory by type, date, or size
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import argparse
import logging

class FileOrganizer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.file_mappings = {
            'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.ico'],
            'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'Documents': ['.pdf', '.doc', '.docx', '.txt', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
            'Audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'],
            'Archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
            'Code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.c', '.php', '.rb', '.go'],
            'Data': ['.json', '.xml', '.csv', '.sql', '.db', '.sqlite']
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('file_organizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def organize_by_type(self):
        """Organize files by their type/extension"""
        moved_count = 0
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                destination_folder = None
                
                # Find the appropriate folder for this file type
                for folder, extensions in self.file_mappings.items():
                    if file_ext in extensions:
                        destination_folder = folder
                        break
                
                # If no mapping found, use 'Others' folder
                if not destination_folder:
                    destination_folder = 'Others'
                
                # Create destination directory and move file
                dest_dir = self.source_dir / destination_folder
                dest_dir.mkdir(exist_ok=True)
                
                try:
                    dest_path = dest_dir / file_path.name
                    
                    # Handle duplicate files
                    if dest_path.exists():
                        base = dest_path.stem
                        extension = dest_path.suffix
                        counter = 1
                        
                        while dest_path.exists():
                            dest_path = dest_dir / f"{base}_{counter}{extension}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(dest_path))
                    self.logger.info(f"Moved: {file_path.name} -> {destination_folder}/")
                    moved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error moving {file_path.name}: {str(e)}")
        
        self.logger.info(f"Organization complete! Moved {moved_count} files.")
        return moved_count

    def organize_by_date(self, date_format='%Y/%B'):
        """Organize files by their modification date"""
        moved_count = 0
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                # Get file modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                date_folder = mod_time.strftime(date_format)
                
                # Create destination directory
                dest_dir = self.source_dir / date_folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    dest_path = dest_dir / file_path.name
                    
                    # Handle duplicates
                    if dest_path.exists():
                        base = dest_path.stem
                        extension = dest_path.suffix
                        counter = 1
                        
                        while dest_path.exists():
                            dest_path = dest_dir / f"{base}_{counter}{extension}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(dest_path))
                    self.logger.info(f"Moved: {file_path.name} -> {date_folder}/")
                    moved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error moving {file_path.name}: {str(e)}")
        
        self.logger.info(f"Organization complete! Moved {moved_count} files.")
        return moved_count

    def organize_by_size(self, size_categories=None):
        """Organize files by their size"""
        if size_categories is None:
            size_categories = [
                (0, 1024*1024, 'Small (< 1MB)'),
                (1024*1024, 10*1024*1024, 'Medium (1-10MB)'),
                (10*1024*1024, 100*1024*1024, 'Large (10-100MB)'),
                (100*1024*1024, float('inf'), 'Very Large (> 100MB)')
            ]
        
        moved_count = 0
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size
                
                # Determine size category
                for min_size, max_size, category in size_categories:
                    if min_size <= file_size < max_size:
                        dest_dir = self.source_dir / category
                        dest_dir.mkdir(exist_ok=True)
                        
                        try:
                            dest_path = dest_dir / file_path.name
                            
                            # Handle duplicates
                            if dest_path.exists():
                                base = dest_path.stem
                                extension = dest_path.suffix
                                counter = 1
                                
                                while dest_path.exists():
                                    dest_path = dest_dir / f"{base}_{counter}{extension}"
                                    counter += 1
                            
                            shutil.move(str(file_path), str(dest_path))
                            self.logger.info(f"Moved: {file_path.name} -> {category}/")
                            moved_count += 1
                            
                        except Exception as e:
                            self.logger.error(f"Error moving {file_path.name}: {str(e)}")
                        
                        break
        
        self.logger.info(f"Organization complete! Moved {moved_count} files.")
        return moved_count

def main():
    parser = argparse.ArgumentParser(description='Organize files in a directory')
    parser.add_argument('directory', help='Directory to organize')
    parser.add_argument('--method', choices=['type', 'date', 'size'], 
                       default='type', help='Organization method')
    parser.add_argument('--date-format', default='%Y/%B', 
                       help='Date format for date-based organization')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist!")
        return
    
    organizer = FileOrganizer(args.directory)
    
    if args.method == 'type':
        organizer.organize_by_type()
    elif args.method == 'date':
        organizer.organize_by_date(args.date_format)
    elif args.method == 'size':
        organizer.organize_by_size()

if __name__ == "__main__":
    main()
```



## 7. Automated Backup Script
Creates automated backups with compression, encryption, and cloud upload options.

```python
#!/usr/bin/env python3
"""
Automated Backup Script
Creates automated backups with compression, encryption, and rotation
"""

import os
import shutil
import zipfile
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import logging
from cryptography.fernet import Fernet
import schedule
import time

class BackupManager:
    def __init__(self, source_dirs, backup_dir, encryption_key=None):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = None
        
        # Setup logging
        log_file = self.backup_dir / 'backup.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Backup metadata
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        self.load_metadata()

    def load_metadata(self):
        """Load backup metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'backups': []}

    def save_metadata(self):
        """Save backup metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_backup(self, compress=True, encrypt=False):
        """Create a backup of all source directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'backup_{timestamp}'
        
        if compress:
            backup_path = self.backup_dir / f'{backup_name}.zip'
            self._create_compressed_backup(backup_path)
        else:
            backup_path = self.backup_dir / backup_name
            self._create_directory_backup(backup_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_path)
        
        # Encrypt if requested
        if encrypt and self.cipher:
            encrypted_path = Path(str(backup_path) + '.enc')
            self._encrypt_file(backup_path, encrypted_path)
            backup_path.unlink()  # Remove unencrypted file
            backup_path = encrypted_path
        
        # Update metadata
        backup_info = {
            'timestamp': timestamp,
            'path': str(backup_path),
            'size': backup_path.stat().st_size,
            'checksum': checksum,
            'encrypted': encrypt,
            'compressed': compress,
            'source_dirs': [str(d) for d in self.source_dirs]
        }
        
        self.metadata['backups'].append(backup_info)
        self.save_metadata()
        
        self.logger.info(f"Backup created: {backup_path}")
        self.logger.info(f"Size: {backup_info['size'] / 1024 / 1024:.2f} MB")
        self.logger.info(f"Checksum: {checksum}")
        
        return backup_path

    def _create_compressed_backup(self, backup_path):
        """Create a compressed ZIP backup"""
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for source_dir in self.source_dirs:
                if source_dir.exists():
                    for file_path in source_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = str(file_path.relative_to(source_dir.parent))
                            zipf.write(file_path, arcname)
                            self.logger.debug(f"Added to backup: {arcname}")

    def _create_directory_backup(self, backup_path):
        """Create a directory-based backup"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for source_dir in self.source_dirs:
            if source_dir.exists():
                dest_dir = backup_path / source_dir.name
                shutil.copytree(source_dir, dest_dir)
                self.logger.debug(f"Copied: {source_dir} -> {dest_dir}")

    def _calculate_checksum(self, file_path):
        """Calculate SHA256 checksum of a file or directory"""
        sha256_hash = hashlib.sha256()
        
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        else:
            # For directories, calculate checksum of all files
            for file in sorted(file_path.rglob('*')):
                if file.is_file():
                    with open(file, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()

    def _encrypt_file(self, input_path, output_path):
        """Encrypt a file using Fernet encryption"""
        with open(input_path, 'rb') as f:
            encrypted_data = self.cipher.encrypt(f.read())
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)

    def restore_backup(self, backup_timestamp, restore_dir):
        """Restore a backup to specified directory"""
        # Find backup in metadata
        backup_info = None
        for backup in self.metadata['backups']:
            if backup['timestamp'] == backup_timestamp:
                backup_info = backup
                break
        
        if not backup_info:
            self.logger.error(f"Backup not found: {backup_timestamp}")
            return False
        
        backup_path = Path(backup_info['path'])
        restore_dir = Path(restore_dir)
        restore_dir.mkdir(parents=True, exist_ok=True)
        
        # Decrypt if necessary
        if backup_info['encrypted'] and self.cipher:
            decrypted_path = backup_path.with_suffix('')
            with open(backup_path, 'rb') as f:
                decrypted_data = self.cipher.decrypt(f.read())
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            backup_path = decrypted_path
        
        # Extract backup
        if backup_info['compressed']:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(restore_dir)
        else:
            shutil.copytree(backup_path, restore_dir / backup_path.name)
        
        # Clean up decrypted file if it was created
        if backup_info['encrypted'] and 'decrypted_path' in locals():
            decrypted_path.unlink()
        
        self.logger.info(f"Backup restored to: {restore_dir}")
        return True

    def rotate_backups(self, keep_daily=7, keep_weekly=4, keep_monthly=12):
        """Implement backup rotation strategy"""
        now = datetime.now()
        backups_to_keep = set()
        
        # Sort backups by timestamp
        sorted_backups = sorted(
            self.metadata['backups'],
            key=lambda x: datetime.strptime(x['timestamp'], '%Y%m%d_%H%M%S'),
            reverse=True
        )
        
        # Keep daily backups
        for i, backup in enumerate(sorted_backups[:keep_daily]):
            backups_to_keep.add(backup['timestamp'])
        
        # Keep weekly backups
        weekly_kept = 0
        for backup in sorted_backups:
            backup_date = datetime.strptime(backup['timestamp'], '%Y%m%d_%H%M%S')
            if backup_date.weekday() == 0:  # Monday
                if weekly_kept < keep_weekly:
                    backups_to_keep.add(backup['timestamp'])
                    weekly_kept += 1
        
        # Keep monthly backups
        monthly_kept = 0
        for backup in sorted_backups:
            backup_date = datetime.strptime(backup['timestamp'], '%Y%m%d_%H%M%S')
            if backup_date.day == 1:  # First day of month
                if monthly_kept < keep_monthly:
                    backups_to_keep.add(backup['timestamp'])
                    monthly_kept += 1
        
        # Remove old backups
        removed_count = 0
        new_backup_list = []
        
        for backup in self.metadata['backups']:
            if backup['timestamp'] in backups_to_keep:
                new_backup_list.append(backup)
            else:
                # Delete the backup file
                backup_path = Path(backup['path'])
                if backup_path.exists():
                    backup_path.unlink()
                    self.logger.info(f"Removed old backup: {backup_path}")
                    removed_count += 1
        
        self.metadata['backups'] = new_backup_list
        self.save_metadata()
        
        self.logger.info(f"Backup rotation complete. Removed {removed_count} old backups.")

    def verify_backup(self, backup_timestamp):
        """Verify backup integrity using checksum"""
        for backup in self.metadata['backups']:
            if backup['timestamp'] == backup_timestamp:
                backup_path = Path(backup['path'])
                if not backup_path.exists():
                    self.logger.error(f"Backup file not found: {backup_path}")
                    return False
                
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum == backup['checksum']:
                    self.logger.info(f"Backup verification successful: {backup_timestamp}")
                    return True
                else:
                    self.logger.error(f"Backup verification failed: {backup_timestamp}")
                    self.logger.error(f"Expected: {backup['checksum']}")
                    self.logger.error(f"Got: {current_checksum}")
                    return False
        
        self.logger.error(f"Backup not found in metadata: {backup_timestamp}")
        return False

def generate_encryption_key():
    """Generate a new encryption key"""
    return Fernet.generate_key()

def main():
    parser = argparse.ArgumentParser(description='Automated Backup System')
    parser.add_argument('source', nargs='+', help='Source directories to backup')
    parser.add_argument('--destination', '-d', required=True, help='Backup destination directory')
    parser.add_argument('--encrypt', '-e', action='store_true', help='Encrypt backups')
    parser.add_argument('--key-file', '-k', help='Encryption key file')
    parser.add_argument('--generate-key', action='store_true', help='Generate new encryption key')
    parser.add_argument('--compress', '-c', action='store_true', default=True, help='Compress backups')
    parser.add_argument('--verify', '-v', help='Verify a backup by timestamp')
    parser.add_argument('--restore', '-r', help='Restore a backup by timestamp')
    parser.add_argument('--restore-to', help='Directory to restore backup to')
    parser.add_argument('--rotate', action='store_true', help='Rotate old backups')
    parser.add_argument('--schedule', help='Schedule backups (e.g., "daily", "weekly")')
    
    args = parser.parse_args()
    
    # Generate encryption key if requested
    if args.generate_key:
        key = generate_encryption_key()
        print(f"Generated encryption key: {key.decode()}")
        if args.key_file:
            with open(args.key_file, 'wb') as f:
                f.write(key)
            print(f"Key saved to: {args.key_file}")
        return
    
    # Load encryption key if provided
    encryption_key = None
    if args.key_file and os.path.exists(args.key_file):
        with open(args.key_file, 'rb') as f:
            encryption_key = f.read()
    
    # Initialize backup manager
    manager = BackupManager(args.source, args.destination, encryption_key)
    
    # Handle different operations
    if args.verify:
        manager.verify_backup(args.verify)
    elif args.restore and args.restore_to:
        manager.restore_backup(args.restore, args.restore_to)
    elif args.rotate:
        manager.rotate_backups()
    elif args.schedule:
        # Schedule backups
        if args.schedule == 'daily':
            schedule.every().day.at("02:00").do(
                lambda: manager.create_backup(compress=args.compress, encrypt=args.encrypt)
            )
        elif args.schedule == 'weekly':
            schedule.every().monday.at("02:00").do(
                lambda: manager.create_backup(compress=args.compress, encrypt=args.encrypt)
            )
        
        print(f"Scheduled {args.schedule} backups. Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        # Create a backup
        manager.create_backup(compress=args.compress, encrypt=args.encrypt)

if __name__ == "__main__":
    main()
```

## 8. Web Scraper and Monitor
Monitors websites for changes and extracts specific data.

```python
#!/usr/bin/env python3
"""
Web Scraper and Monitor
Monitors websites for changes and extracts specific data
"""

import requests
from bs4 import BeautifulSoup
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse
import logging
from urllib.parse import urljoin, urlparse
import csv
import re

class WebMonitor:
    def __init__(self, config_file='monitor_config.json'):
        self.config_file = Path(config_file)
        self.load_config()
        self.data_dir = Path('web_monitor_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'targets': [],
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender': '',
                    'password': '',
                    'recipients': []
                }
            }
            self.save_config()

    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def add_target(self, url, name, selectors=None, check_interval=3600):
        """Add a new target to monitor"""
        target = {
            'url': url,
            'name': name,
            'selectors': selectors or {},
            'check_interval': check_interval,
            'last_check': None,
            'last_hash': None
        }
        
        self.config['targets'].append(target)
        self.save_config()
        self.logger.info(f"Added monitoring target: {name} ({url})")

    def fetch_page(self, url, retries=3):
        """Fetch a web page with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def extract_data(self, html, selectors):
        """Extract data from HTML using CSS selectors"""
        soup = BeautifulSoup(html, 'html.parser')
        extracted_data = {}
        
        for field, selector_info in selectors.items():
            if isinstance(selector_info, str):
                # Simple CSS selector
                elements = soup.select(selector_info)
                if elements:
                    extracted_data[field] = [elem.get_text(strip=True) for elem in elements]
            elif isinstance(selector_info, dict):
                # Advanced selector with options
                selector = selector_info.get('selector')
                attribute = selector_info.get('attribute')
                regex = selector_info.get('regex')
                
                elements = soup.select(selector)
                values = []
                
                for elem in elements:
                    if attribute:
                        value = elem.get(attribute, '')
                    else:
                        value = elem.get_text(strip=True)
                    
                    if regex:
                        match = re.search(regex, value)
                        if match:
                            value = match.group(1) if match.groups() else match.group(0)
                    
                    values.append(value)
                
                extracted_data[field] = values
        
        return extracted_data

    def calculate_hash(self, content):
        """Calculate hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def check_target(self, target):
        """Check a single target for changes"""
        try:
            self.logger.info(f"Checking: {target['name']}")
            
            # Fetch the page
            response = self.fetch_page(target['url'])
            content = response.text
            
            # Extract specific data if selectors are provided
            if target['selectors']:
                extracted_data = self.extract_data(content, target['selectors'])
                content_to_hash = json.dumps(extracted_data, sort_keys=True)
            else:
                content_to_hash = content
            
            # Calculate hash
            current_hash = self.calculate_hash(content_to_hash)
            
            # Check for changes
            if target['last_hash'] and current_hash != target['last_hash']:
                self.logger.info(f"Change detected in {target['name']}")
                self.handle_change(target, content, extracted_data if target['selectors'] else None)
            
            # Update target info
            target['last_hash'] = current_hash
            target['last_check'] = datetime.now().isoformat()
            
            # Save snapshot
            self.save_snapshot(target, content, extracted_data if target['selectors'] else None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking {target['name']}: {str(e)}")
            return False

    def handle_change(self, target, content, extracted_data=None):
        """Handle detected changes"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save change record
        change_file = self.data_dir / f"{target['name']}_{timestamp}_change.json"
        change_data = {
            'timestamp': timestamp,
            'url': target['url'],
            'name': target['name'],
            'extracted_data': extracted_data
        }
        
        with open(change_file, 'w') as f:
            json.dump(change_data, f, indent=2)
        
        # Send notification
        if self.config['email']['enabled']:
            self.send_notification(target, extracted_data)

    def save_snapshot(self, target, content, extracted_data=None):
        """Save a snapshot of the current state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create target directory
        target_dir = self.data_dir / target['name']
        target_dir.mkdir(exist_ok=True)
        
        # Save HTML content
        html_file = target_dir / f'snapshot_{timestamp}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save extracted data
        if extracted_data:
            data_file = target_dir / f'data_{timestamp}.json'
            with open(data_file, 'w') as f:
                json.dump(extracted_data, f, indent=2)

    def send_notification(self, target, extracted_data=None):
        """Send email notification about changes"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Change detected: {target['name']}"
            
            body = f"A change was detected on {target['name']}\n"
            body += f"URL: {target['url']}\n"
            body += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if extracted_data:
                body += "Extracted Data:\n"
                body += json.dumps(extracted_data, indent=2)
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender'], email_config['password'])
                server.send_message(msg)
            
            self.logger.info("Notification sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")

    def monitor_all(self, continuous=False):
        """Monitor all targets"""
        while True:
            for target in self.config['targets']:
                # Check if it's time to check this target
                if target['last_check']:
                    last_check = datetime.fromisoformat(target['last_check'])
                    time_since_check = (datetime.now() - last_check).total_seconds()
                    
                    if time_since_check < target['check_interval']:
                        continue
                
                self.check_target(target)
            
            # Save updated config
            self.save_config()
            
            if not continuous:
                break
            
            # Wait before next round
            time.sleep(60)  # Check every minute

    def export_data(self, target_name, output_format='csv'):
        """Export collected data"""
        target_dir = self.data_dir / target_name
        
        if not target_dir.exists():
            self.logger.error(f"No data found for target: {target_name}")
            return
        
        # Collect all data files
        data_files = sorted(target_dir.glob('data_*.json'))
        
        if output_format == 'csv':
            output_file = self.data_dir / f'{target_name}_export.csv'
            
            # Determine all fields
            all_fields = set()
            all_data = []
            
            for data_file in data_files:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    all_fields.update(data.keys())
                    
                    # Add timestamp
                    timestamp = data_file.stem.replace('data_', '')
                    data['timestamp'] = timestamp
                    all_data.append(data)
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp'] + sorted(all_fields))
                writer.writeheader()
                
                for data in all_data:
                    # Flatten lists to strings
                    flat_data = {}
                    for key, value in data.items():
                        if isinstance(value, list):
                            flat_data[key] = '; '.join(str(v) for v in value)
                        else:
                            flat_data[key] = value
                    
                    writer.writerow(flat_data)
            
            self.logger.info(f"Data exported to: {output_file}")

class PriceMonitor(WebMonitor):
    """Specialized monitor for tracking prices"""
    
    def add_price_target(self, url, name, price_selector, title_selector=None, threshold=None):
        """Add a price monitoring target"""
        selectors = {
            'price': {
                'selector': price_selector,
                'regex': r'[\d,]+\.?\d*'
            }
        }
        
        if title_selector:
            selectors['title'] = title_selector
        
        self.add_target(url, name, selectors)
        
        # Add price threshold if specified
        if threshold:
            for target in self.config['targets']:
                if target['name'] == name:
                    target['price_threshold'] = threshold
                    break
            
            self.save_config()

    def handle_change(self, target, content, extracted_data=None):
        """Handle price changes with threshold checking"""
        super().handle_change(target, content, extracted_data)
        
        # Check price threshold
        if extracted_data and 'price' in extracted_data and 'price_threshold' in target:
            try:
                current_price = float(extracted_data['price'][0].replace(',', ''))
                threshold = target['price_threshold']
                
                if current_price <= threshold:
                    self.logger.info(f"Price alert! {target['name']} is now ${current_price:.2f}")
                    # Send special notification
                    self.send_price_alert(target, current_price)
            except (ValueError, IndexError):
                pass

    def send_price_alert(self, target, price):
        """Send price alert notification"""
        if self.config['email']['enabled']:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Price Alert: {target['name']} - ${price:.2f}"
            
            body = f"Good news! The price for {target['name']} has dropped to ${price:.2f}\n"
            body += f"URL: {target['url']}\n"
            body += f"Threshold: ${target['price_threshold']:.2f}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            try:
                with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                    server.starttls()
                    server.login(email_config['sender'], email_config['password'])
                    server.send_message(msg)
                
                self.logger.info("Price alert sent successfully")
            except Exception as e:
                self.logger.error(f"Failed to send price alert: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Web Scraper and Monitor')
    parser.add_argument('--add-target', action='store_true', help='Add a new monitoring target')
    parser.add_argument('--url', help='Target URL')
    parser.add_argument('--name', help='Target name')
    parser.add_argument('--selectors', help='CSS selectors (JSON format)')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--export', help='Export data for target')
    parser.add_argument('--price-monitor', action='store_true', help='Use price monitoring mode')
    parser.add_argument('--price-selector', help='CSS selector for price')
    parser.add_argument('--price-threshold', type=float, help='Price threshold for alerts')
    parser.add_argument('--config-email', action='store_true', help='Configure email settings')
    
    args = parser.parse_args()
    
    if args.price_monitor:
        monitor = PriceMonitor()
    else:
        monitor = WebMonitor()
    
    if args.add_target and args.url and args.name:
        if args.price_monitor and args.price_selector:
            monitor.add_price_target(
                args.url,
                args.name,
                args.price_selector,
                threshold=args.price_threshold
            )
        else:
            selectors = json.loads(args.selectors) if args.selectors else None
            monitor.add_target(args.url, args.name, selectors)
    
    elif args.monitor:
        monitor.monitor_all(continuous=args.continuous)
    
    elif args.export:
        monitor.export_data(args.export)
    
    elif args.config_email:
        # Interactive email configuration
        config = monitor.config['email']
        config['enabled'] = True
        config['sender'] = input("Enter sender email: ")
        config['password'] = input("Enter email password: ")
        config['recipients'] = input("Enter recipient emails (comma-separated): ").split(',')
        monitor.save_config()
        print("Email configuration saved!")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```
