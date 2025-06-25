
# 20-Hour Python Programming Course: Complete Lesson Plan

## Course Overview
**Target Audience:** Beginners with no programming experience  
**Duration:** 20 hours (10 x 2-hour sessions or 20 x 1-hour sessions)  
**Goal:** Build practical Python skills for data analysis, automation, and problem-solving  

---

## Hour 1-2: Python Foundations & Setup
### Learning Objectives
- Understand Python's applications and popularity
- Set up development environment
- Write and execute first programs
- Master basic syntax rules

### Content
```python
# Hello World Example
print("Hello, World!") 
print(2 + 3)  # Basic arithmetic

# Variables and Input
name = input("What's your name? ")
print(f"Welcome, {name}!")
```

### Exercises
1. Create a temperature converter (Fahrenheit to Celsius)
2. Build a simple tip calculator

---

## Hour 3-4: Data Types & Strings
### Learning Objectives
- Work with core data types (int, float, str, bool)
- Master string manipulation
- Handle type conversions

### Key Concepts
```python
# String Methods
text = "Python Programming"
print(text.upper())  # 'PYTHON PROGRAMMING'

# Type Conversion
age = int("25")  # String to integer
```

### Project
Build a text analyzer that counts characters, words, and sentences.

---

## Hour 5-6: Lists & Data Collections
### Learning Objectives
- Create and manipulate lists
- Understand slicing and indexing
- Use list methods effectively

### Examples
```python
# List Operations
fruits = ["apple", "banana"]
fruits.append("orange")  # Add item
print(fruits[1:])  # Slicing
```

### Application
Develop a to-do list manager with add/remove functionality.

---

## Hour 7-8: Dictionaries & Data Structures
### Learning Objectives
- Work with key-value pairs
- Compare lists vs dictionaries
- Organize complex data

### Implementation
```python
# Dictionary Example
student = {
    "name": "Alice",
    "grades": [85, 90]
}
print(student["name"])
```

### Project
Create a contact book that stores names, emails, and phone numbers.

---

## Hour 9-10: Control Flow
### Learning Objectives
- Implement if/elif/else logic
- Use comparison operators
- Handle multiple conditions

### Code Sample
```python
# Grade Calculator
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
```

### Exercise
Build a number guessing game with hints.

---

## Hour 11-12: Loops
### Learning Objectives
- Master for and while loops
- Avoid infinite loops
- Process collections efficiently

### Examples
```python
# For Loop
for i in range(5):
    print(i)

# While Loop
count = 0
while count < 3:
    print(count)
    count += 1
```

### Project
Develop a multiplication table generator.

---

## Hour 13-14: Functions
### Learning Objectives
- Create reusable functions
- Use parameters and return values
- Implement lambda functions

### Implementation
```python
# Function Example
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

### Exercise
Build a calculator with add/subtract/multiply functions.

---

## Hour 15-16: File Handling
### Learning Objectives
- Read/write text files
- Work with CSV data
- Store JSON structures

### Code Sample
```python
# File Operations
with open("data.txt", "w") as file:
    file.write("Sample content")

# CSV Handling
import csv
with open("data.csv") as file:
    reader = csv.reader(file)
```

### Project
Create a diary application that saves entries to files.

---

## Hour 17-18: Data Analysis
### Learning Objectives
- Install pandas and matplotlib
- Analyze datasets
- Create visualizations

### Examples
```python
# Pandas Basics
import pandas as pd
data = pd.read_csv("sales.csv")
print(data.head())
```

### Application
Analyze student performance data and generate reports.

---

## Hour 19-20: Final Project
### Project Options
1. **Personal Finance Manager**
2. **Student Grade System**
3. **Inventory Tracker**

### Example Structure
```python
class FinanceManager:
    def __init__(self):
        self.transactions = []
    
    def add_transaction(self, amount, category):
        self.transactions.append({
            "amount": amount,
            "category": category
        })
```

### Review Checklist
- [ ] Code works as intended
- [ ] Proper error handling
- [ ] Clean documentation
- [ ] Efficient logic

---

## Next Steps
1. Learn web development with Flask/Django
2. Explore data science with pandas
3. Study algorithms and OOP
4. Contribute to open source projects

## Certification Requirements
- Complete all exercises
- Submit final project
- Pass code review
