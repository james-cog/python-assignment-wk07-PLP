# Advanced Data Analysis with Pandas and Matplotlib

## 📊 Project Overview

This project demonstrates comprehensive data analysis techniques using Python's most popular data science libraries. It implements all the required functionality for analyzing CSV datasets and creating beautiful visualizations.

## 🎯 Features Implemented

### Task 1: Load and Explore Dataset
- ✅ Load CSV datasets with error handling
- ✅ Display data using `.head()` method
- ✅ Explore dataset structure (data types, shape, missing values)
- ✅ Clean datasets by filling/dropping missing values
- ✅ Works with both CSV files and built-in datasets (Iris)

### Task 2: Basic Data Analysis  
- ✅ Compute descriptive statistics using `.describe()`
- ✅ Perform groupby analysis on categorical columns
- ✅ Calculate mean/median/std for numerical columns per group
- ✅ Pattern identification and insights discovery

### Task 3: Data Visualization
- ✅ **Line Chart**: Trends over time analysis
- ✅ **Bar Chart**: Comparison across categories
- ✅ **Histogram**: Distribution analysis of numerical data
- ✅ **Scatter Plot**: Relationship visualization between variables
- ✅ **Enhanced customization**: Titles, labels, legends, grid styling
- ✅ **Seaborn integration**: Beautiful plotting styles

### Additional Features
- 🔧 **Robust Error Handling**: File reading, missing data, network issues
- 📈 **Multiple Dataset Support**: CSV files or sklearn datasets
- 📊 **Comprehensive Visualization**: All 4 required chart types
- 🎨 **Professional Styling**: Enhanced with matplotlib and seaborn
- 📄 **Analysis Reports**: Automatic report generation
- 🔧 **Dependency Management**: Clear installation guidance

## 🚀 Installation

### Prerequisites
Make sure you have Python 3.7+ installed.

### Install Dependencies
```bash
pip install pandas matplotlib seaborn numpy scikit-learn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## 📋 Usage

### Basic Usage
```python
python data_analysis_project.py
```

### How it Works
1. **Auto-detects** CSV files in your directory
2. **Loads and explores** the dataset automatically
3. **Cleans data** (handles missing values)
4. **Performs statistical analysis**
5. **Creates 4 custom visualizations**
6. **Generates analysis report**
7. **Saves results** to timestamped files

## 📊 Supported Dataset Types

### CSV Files
- Any CSV with numerical and categorical columns
- Handles missing data automatically
- Flexible column detection

### Built-in Datasets
- **Iris Dataset**: Classic machine learning dataset
- **Wine Dataset**: Available for testing purposes

### CSV File Requirements
Your CSV should ideally contain:
- **Categorical columns**: For groupby analysis (like "species", "region", "category")
- **Numerical columns**: For statistical analysis and visualizations
- Any format: UTF-8, different separators supported by pandas

## 🎨 Visualization Details

### 1. Line Chart
- Shows trends over simulated time periods
- Ideal for time-series data analysis

### 2. Bar Chart  
- Compares values across categories
- Displays average values per group
- Color-coded and value-labeled

### 3. Histogram
- Distribution analysis of numerical columns
- Mean line overlay for reference
- Frequency analysis

### 4. Scatter Plot
- Relationship between two numerical variables  
- Color-coded by categorical grouping (if available)
- Trend line overlay

## 🔧 Error Handling

The script handles:
- **File Reading Errors**: Missing files, permission issues
- **Data Type Issues**: Automatic type detection and conversion  
- **Missing Data**: Intelligent filling strategies
- **Dependency Issues**: Clear installation guidance
- **Visualization Errors**: Graceful fallbacks

## 📄 Output Files

- **Console Output**: Real-time analysis progress
- **Plots**: Displayed interactively with matplotlib
- **Report File**: `data_analysis_report_YYYYMMDD_HHMMSS.txt`

## 🎯 Project Structure

```
project/
├── data_analysis_project.py    # Main analysis script
├── requirements.txt           # Dependencies list
├── README.md                 # This documentation
└── *.csv                     # Your data files (optional)
```

## 🔍 Example Analysis Features

### For Iris Dataset:
```
📊 First 5 rows inspection
📈 Descriptive statistics (mean, std, min, max)
🔍 Species-based groupby analysis  
📊 Distribution visualizations
📈 Relationship between petal/sepal measurements
```

### For CSV Sales Data:
```
📊 Store data loading and cleaning
📈 Regional sales comparison analysis  
📊 Customer demographics exploration
📈 Time-series sales trends
```

## ✨ Advanced Features

- **Dynamic Column Detection**: Automatically identifies numerical/categorical columns
- **Smart Data Cleaning**: Fills numerical with median, categorical with mode
- **Professional Styling**: Enhanced with seaborn and matplotlib customization
- **Pattern Analysis**: Automatic insights generation
- **Report Generation**: Timestamped analysis reports
- **Cross-platform Compatible**: Works on Windows, Mac, Linux

## 🚨 Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Install dependencies with `pip install pandas matplotlib seaborn numpy scikit-learn`
2. **No CSV files found**: Script will use Iris dataset as demo
3. **Visualization not showing**: Ensure matplotlib backend is configured
4. **Permission errors**: Check file permissions for report writing

### Support:
- Check requirements.txt for exact versions
- Ensure Python 3.7+ environment
- Verify matplotlib backend configuration

---

🎯 **This project fulfills all requirements for comprehensive data analysis using Pandas and Matplotlib for visualization!** 📊✨
