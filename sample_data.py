"""
Sample Data Generator

This module creates sample datasets for demonstrating the AI tabular data agent.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_sales_data(num_records: int = 1000) -> pd.DataFrame:
    """
    Create a sample sales dataset.
    
    Args:
        num_records (int): Number of records to generate
        
    Returns:
        pd.DataFrame: Sample sales data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate data
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    sales_reps = [f'Rep_{i:03d}' for i in range(1, 51)]
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        record = {
            'order_id': f'ORD_{i+1:06d}',
            'date': start_date + timedelta(days=random.randint(0, 365)),
            'region': random.choice(regions),
            'product': random.choice(products),
            'sales_rep': random.choice(sales_reps),
            'quantity': random.randint(1, 100),
            'unit_price': round(random.uniform(10, 500), 2),
            'discount': round(random.uniform(0, 0.3), 2),
            'customer_type': random.choice(['New', 'Existing', 'Premium'])
        }
        
        # Calculate derived fields
        record['gross_revenue'] = record['quantity'] * record['unit_price']
        record['discount_amount'] = record['gross_revenue'] * record['discount']
        record['net_revenue'] = record['gross_revenue'] - record['discount_amount']
        
        data.append(record)
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def create_employee_data(num_records: int = 500) -> pd.DataFrame:
    """
    Create a sample employee dataset.
    
    Args:
        num_records (int): Number of records to generate
        
    Returns:
        pd.DataFrame: Sample employee data
    """
    np.random.seed(42)
    random.seed(42)
    
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    positions = {
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
        'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Content Creator', 'SEO Specialist'],
        'Sales': ['Sales Rep', 'Account Manager', 'Sales Manager', 'Business Development'],
        'HR': ['HR Specialist', 'Recruiter', 'HR Manager', 'Training Coordinator'],
        'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'Controller'],
        'Operations': ['Operations Specialist', 'Project Manager', 'Operations Manager', 'Coordinator']
    }
    
    first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack',
                   'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Paul', 'Quinn', 'Ruby', 'Sam', 'Tina']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']
    
    data = []
    
    for i in range(num_records):
        department = random.choice(departments)
        position = random.choice(positions[department])
        
        # Generate realistic salary based on department and position
        base_salary = {
            'Engineering': 80000,
            'Marketing': 60000,
            'Sales': 55000,
            'HR': 50000,
            'Finance': 65000,
            'Operations': 55000
        }[department]
        
        position_multiplier = {
            'Specialist': 1.0,
            'Rep': 1.0,
            'Analyst': 1.1,
            'Creator': 1.0,
            'Coordinator': 1.0,
            'Engineer': 1.2,
            'Senior': 1.4,
            'Manager': 1.6,
            'Lead': 1.5,
            'Controller': 1.8
        }
        
        multiplier = 1.0
        for key, mult in position_multiplier.items():
            if key in position:
                multiplier = mult
                break
        
        salary = int(base_salary * multiplier * random.uniform(0.8, 1.3))
        
        record = {
            'employee_id': f'EMP_{i+1:04d}',
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'department': department,
            'position': position,
            'hire_date': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)),
            'salary': salary,
            'age': random.randint(22, 65),
            'years_experience': random.randint(0, 20),
            'performance_rating': round(random.uniform(2.5, 5.0), 1),
            'remote_work': random.choice([True, False])
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    df['hire_date'] = pd.to_datetime(df['hire_date'])
    df['full_name'] = df['first_name'] + ' ' + df['last_name']
    
    return df

def create_customer_data(num_records: int = 800) -> pd.DataFrame:
    """
    Create a sample customer dataset.
    
    Args:
        num_records (int): Number of records to generate
        
    Returns:
        pd.DataFrame: Sample customer data
    """
    np.random.seed(42)
    random.seed(42)
    
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education']
    company_sizes = ['Small (1-50)', 'Medium (51-200)', 'Large (201-1000)', 'Enterprise (1000+)']
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan']
    
    data = []
    
    for i in range(num_records):
        record = {
            'customer_id': f'CUST_{i+1:05d}',
            'company_name': f'Company {chr(65 + i % 26)}{i+1}',
            'industry': random.choice(industries),
            'company_size': random.choice(company_sizes),
            'country': random.choice(countries),
            'registration_date': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)),
            'total_orders': random.randint(1, 50),
            'total_revenue': round(random.uniform(1000, 100000), 2),
            'avg_order_value': 0,  # Will calculate
            'last_order_date': datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)),
            'customer_satisfaction': round(random.uniform(1, 5), 1),
            'is_active': random.choice([True, False]),
            'support_tickets': random.randint(0, 20)
        }
        
        # Calculate average order value
        record['avg_order_value'] = round(record['total_revenue'] / record['total_orders'], 2)
        
        data.append(record)
    
    df = pd.DataFrame(data)
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    
    return df

def create_product_data(num_records: int = 200) -> pd.DataFrame:
    """
    Create a sample product dataset.
    
    Args:
        num_records (int): Number of records to generate
        
    Returns:
        pd.DataFrame: Sample product data
    """
    np.random.seed(42)
    random.seed(42)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
    brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']
    
    data = []
    
    for i in range(num_records):
        category = random.choice(categories)
        
        # Category-specific price ranges
        price_ranges = {
            'Electronics': (50, 2000),
            'Clothing': (20, 300),
            'Home & Garden': (15, 500),
            'Sports': (25, 800),
            'Books': (10, 50),
            'Toys': (5, 150)
        }
        
        min_price, max_price = price_ranges[category]
        
        record = {
            'product_id': f'PROD_{i+1:04d}',
            'product_name': f'{category} Product {i+1}',
            'category': category,
            'brand': random.choice(brands),
            'price': round(random.uniform(min_price, max_price), 2),
            'cost': 0,  # Will calculate
            'stock_quantity': random.randint(0, 1000),
            'reorder_level': random.randint(10, 100),
            'supplier_rating': round(random.uniform(1, 5), 1),
            'launch_date': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)),
            'is_discontinued': random.choice([True, False]),
            'weight_kg': round(random.uniform(0.1, 50), 2),
            'rating': round(random.uniform(1, 5), 1),
            'review_count': random.randint(0, 1000)
        }
        
        # Calculate cost (60-80% of price)
        record['cost'] = round(record['price'] * random.uniform(0.6, 0.8), 2)
        record['profit_margin'] = round((record['price'] - record['cost']) / record['price'] * 100, 1)
        
        data.append(record)
    
    df = pd.DataFrame(data)
    df['launch_date'] = pd.to_datetime(df['launch_date'])
    
    return df

def save_sample_datasets(output_dir: str = "sample_datasets"):
    """
    Generate and save all sample datasets.
    
    Args:
        output_dir (str): Directory to save the datasets
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    datasets = {
        'sales_data.csv': create_sales_data(1000),
        'employee_data.csv': create_employee_data(500),
        'customer_data.csv': create_customer_data(800),
        'product_data.csv': create_product_data(200)
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}: {len(df)} rows, {len(df.columns)} columns")
    
    # Create a README file
    readme_content = """# Sample Datasets for AI Tabular Data Agent

This directory contains sample datasets for demonstrating the AI tabular data agent capabilities.

## Datasets:

### 1. sales_data.csv
- **Description**: Sales transaction data with orders, products, and revenue information
- **Use cases**: Revenue analysis, sales performance, regional comparisons
- **Sample questions**:
  - "What is the total revenue by region?"
  - "Which product has the highest sales?"
  - "Show me monthly sales trends"

### 2. employee_data.csv
- **Description**: Employee information including departments, salaries, and performance
- **Use cases**: HR analytics, salary analysis, performance evaluation
- **Sample questions**:
  - "What is the average salary by department?"
  - "How many employees work remotely?"
  - "Show me the distribution of performance ratings"

### 3. customer_data.csv
- **Description**: Customer information with company details and transaction history
- **Use cases**: Customer analysis, retention studies, market segmentation
- **Sample questions**:
  - "Which industry has the highest average order value?"
  - "How many customers are active?"
  - "What is the customer satisfaction by company size?"

### 4. product_data.csv
- **Description**: Product catalog with pricing, inventory, and performance metrics
- **Use cases**: Inventory management, pricing analysis, product performance
- **Sample questions**:
  - "Which products have low stock levels?"
  - "What is the profit margin by category?"
  - "Show me the top-rated products"

## Getting Started:

1. Upload any of these CSV files to the AI agent interface
2. Ask natural language questions about the data
3. Explore the generated visualizations and insights

## Data Quality:

All datasets are synthetically generated with realistic patterns and relationships.
They include various data types (numeric, categorical, dates) to demonstrate
the full capabilities of the AI agent system.
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nSample datasets saved to '{output_dir}' directory")
    print("README.md created with dataset descriptions and sample questions")

if __name__ == "__main__":
    # Generate and save sample datasets
    save_sample_datasets()
    
    # Display sample data for verification
    print("\n" + "="*50)
    print("SAMPLE DATA PREVIEW")
    print("="*50)
    
    datasets = {
        'Sales Data': create_sales_data(5),
        'Employee Data': create_employee_data(5),
        'Customer Data': create_customer_data(5),
        'Product Data': create_product_data(5)
    }
    
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(df.head())
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

