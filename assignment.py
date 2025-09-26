#!/usr/bin/env python3
"""
Advanced Data Analysis with Pandas and Visualization with Matplotlib
=====================================================================

This project demonstrates comprehensive data analysis techniques using:
- Pandas for data manipulation and analysis
- Matplotlib for custom visualization
- Seaborn for enhanced styling
- Error handling for robust data processing
"""

"""
Advanced Data Analysis with Pandas and Visualization with Matplotlib
=====================================================================

Required dependencies:
    pip install pandas matplotlib seaborn numpy scikit-learn

If dependencies are not installed, the script will guide you through installation.
"""

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.datasets import load_iris
    import warnings
    from datetime import datetime, timedelta
    import os
    import sys
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Set plotting style for better visualizations
    plt.style.use('default')
    try:
        sns.set_palette("husl")
    except:
        pass  # Continue without seaborn styling if not available

except ImportError as e:
    import sys
    print("❌ MISSING DEPENDENCIES")
    print("="*50)
    print("This project requires the following Python packages:")
    print("- pandas")
    print("- matplotlib") 
    print("- seaborn")
    print("- numpy")
    print("- scikit-learn")
    print()
    print("Please install them using:")
    print("pip install pandas matplotlib seaborn numpy scikit-learn")
    print()
    print("Or run: pip install -r requirements.txt")
    print()
    print("Error:", str(e))
    sys.exit(1)

class DataAnalyzer:
    """
    Comprehensive data analysis class with visualization capabilities
    """
    
    def __init__(self):
        self.df = None
        self.original_shape = None
        
    def load_dataset_from_csv(self, filepath):
        """
        Load and explore dataset from CSV file with error handling
        """
        try:
            print("="*60)
            print("📊 LOADING AND EXPLORING DATASET")
            print("="*60)
            
            # Load the CSV file
            self.df = pd.read_csv(filepath)
            self.original_shape = self.df.shape
            
            print(f"✅ Successfully loaded dataset: {filepath}")
            print(f"📏 Original dataset shape: {self.original_shape}")
            
            # Display basic information
            self._explore_dataset()
            self._clean_dataset()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def load_iris_dataset(self):
        """
        Load the famous Iris dataset from scikit-learn
        """
        print("="*60)
        print("📊 LOADING IRIS DATASET (DEMONSTRATION)")
        print("="*60)
        
        # Load Iris dataset
        iris_data = load_iris()
        self.df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
        self.df['species'] = iris_data.target_names[iris_data.target]
        self.original_shape = self.df.shape
        
        print(f"✅ Successfully loaded Iris dataset")
        print(f"📏 Dataset shape: {self.original_shape}")
        
        # Explore the dataset
        self._explore_dataset()
        return True
    
    def _explore_dataset(self):
        """
        Explore dataset structure
        """
        print("\n" + "="*40)
        print("🔍 DATASET EXPLORATION")
        print("="*40)
        
        # Display first 5 rows
        print("\n📋 First 5 rows:")
        print(self.df.head())
        
        # Display info
        print(f"\n📊 Dataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Data types
        print(f"\n🏷️  Data Types:")
        print(self.df.dtypes)
        
        # Missing values
        print(f"\n❓ Missing Values:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found! ✅")
        
        # Basic statistics for numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 Numerical columns found: {list(numeric_cols)}")
    
    def _clean_dataset(self):
        """
        Clean dataset by handling missing values
        """
        print("\n" + "="*40)
        print("🧹 CLEANING DATASET")
        print("="*40)
        
        initial_missing = self.df.isnull().sum().sum()
        
        if initial_missing > 0:
            print(f"Found {initial_missing} missing values")
            
            # Strategy: Fill numerical columns with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].dtype in ['int64', 'float64']:
                    # Fill numerical columns with median
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"✅ Filled missing values in '{col}' with median: {self.df[col].median():.2f}")
                else:
                    # Fill categorical columns with mode
                    mode_value = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_value, inplace=True)
                    print(f"✅ Filled missing values in '{col}' with mode: {mode_value}")
        else:
            print("No missing values found! ✅")
        
        print(f"📊 Final dataset shape: {self.df.shape}")
    
    def basic_statistical_analysis(self):
        """
        Perform basic statistical analysis
        """
        print("\n" + "="*60)
        print("📈 BASIC STATISTICAL ANALYSIS")
        print("="*60)
        
        # Descriptive statistics for numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print("\n📊 Descriptive Statistics for Numerical Columns:")
            print(self.df[numeric_cols].describe().round(3))
            
            # Statistical insights
            print(f"\n🔍 Statistical Insights:")
            for col in numeric_cols:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                print(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        else:
            print("No numerical columns found for statistical analysis")
    
    def groupby_analysis(self):
        """
        Perform groupby analysis on categorical columns
        """
        print("\n" + "="*60)
        print("🔍 GROUPBY ANALYSIS")
        print("="*60)
        
        # Find categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            cat_col = categorical_cols[0]  # Use first categorical column
            num_col = numerical_cols[0]  # Use first numerical column
            
            print(f"📊 Grouping by '{cat_col}' and analyzing '{num_col}':")
            grouped_stats = self.df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count']).round(3)
            print(grouped_stats)
            
            # Additional pattern analysis
            print(f"\n🔍 Pattern Analysis:")
            max_group = grouped_stats['mean'].idxmax()
            min_group = grouped_stats['mean'].idxmin()
            print(f"  Group with highest {num_col}: {max_group} (Mean: {grouped_stats.loc[max_group, 'mean']:.3f})")
            print(f"  Group with lowest {num_col}: {min_group} (Mean: {grouped_stats.loc[min_group, 'mean']:.3f})")
        else:
            print("Need both categorical and numerical columns for groupby analysis")
    
    def create_visualizations(self):
        """
        Create all four required types of visualizations
        """
        print("\n" + "="*60)
        print("📊 CREATING DATA VISUALIZATIONS")
        print("="*60)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Data Analysis Visualizations', fontsize=16, fontweight='bold')
        
        # Get columns for analysis
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # 1. LINE CHART - Trends over time (create simulated time series)
        self._create_line_chart(axes[0, 0])
        
        # 2. BAR CHART - Comparison across categories
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            self._create_bar_chart(axes[0, 1], categorical_cols[0], numerical_cols[0])
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient categorical/numerical data\nfor bar chart', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Bar Chart - Not Applicable', fontweight='bold')
        
        # 3. HISTOGRAM - Distribution of numerical column
        if len(numerical_cols) > 0:
            self._create_histogram(axes[1, 0], numerical_cols[0])
        else:
            axes[1, 0].text(0.5, 0.5, 'No numerical data\nfor histogram', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Histogram - Not Applicable', fontweight='bold')
        
        # 4. SCATTER PLOT - Relationship between two numerical variables
        if len(numerical_cols) >= 2:
            self._create_scatter_plot(axes[1, 1], numerical_cols[0], numerical_cols[1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient numerical variables\nfor scatter plot', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Scatter Plot - Not Applicable', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _create_line_chart(self, ax):
        """
        Create line chart showing trends over time
        """
        # Create simulated time series from first numerical column
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Use first numerical column to simulate time series
            data = self.df[numerical_cols[0]].reset_index()
            data.columns = ['time', 'value']
            
            ax.plot(data.index, data['value'], color='blue', linewidth=2, marker='o', markersize=4)
            ax.set_title(f'Trends in {numerical_cols[0]}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Time Period', fontweight='bold')
            ax.set_ylabel(numerical_cols[0], fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(['Trend Line'])
        else:
            ax.text(0.5, 0.5, 'No numerical data\nfor time series', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Line Chart - Not Applicable', fontweight='bold')
    
    def _create_bar_chart(self, ax, categorical_col, numerical_col):
        """
        Create bar chart showing comparison across categories
        """
        # Group data and calculate mean
        grouped_data = self.df.groupby(categorical_col)[numerical_col].mean().sort_values(ascending=False)
        
        bars = ax.bar(range(len(grouped_data)), grouped_data.values, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(grouped_data))))
        
        ax.set_title(f'Average {numerical_col} by {categorical_col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(categorical_col, fontweight='bold')
        ax.set_ylabel(f'Average {numerical_col}', fontweight='bold')
        ax.set_xticks(range(len(grouped_data)))
        ax.set_xticklabels(grouped_data.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(grouped_data.values):
            ax.text(i, v + 0.01 * (grouped_data.max()), f'{v:.2f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
    
    def _create_histogram(self, ax, numerical_col):
        """
        Create histogram to understand distribution
        """
        ax.hist(self.df[numerical_col], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_title(f'Distribution of {numerical_col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(numerical_col, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_val = self.df[numerical_col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    def _create_scatter_plot(self, ax, x_col, y_col):
        """
        Create scatter plot to show relationship between two numerical variables
        """
        # Color point by categorical variable if available
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            # Use the first categorical column for coloring
            cat_col = categorical_cols[0]
            unique_categories = self.df[cat_col].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
            
            for i, category in enumerate(unique_categories):
                category_data = self.df[self.df[cat_col] == category]
                ax.scatter(category_data[x_col], category_data[y_col], 
                          c=[colors[i]], label=category, alpha=0.7, s=50)
        else:
            ax.scatter(self.df[x_col], self.df[y_col], alpha=0.7, s=50, color='blue')
        
        # Add trend line
        z = np.polyfit(self.df[x_col], self.df[y_col], 1)
        p = np.poly1d(z)
        ax.plot(self.df[x_col], p(self.df[x_col]), "r--", alpha=0.8, linewidth=2)
        
        ax.set_title(f'Relationship: {x_col} vs {y_col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(x_col, fontweight='bold')
        ax.set_ylabel(y_col, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(categorical_cols) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def save_analysis_report(self):
        """
        Save analysis report to file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_analysis_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("DATA ANALYSIS REPORT\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset Shape: {self.df.shape}\n\n")
                
                f.write("COLUMN INFORMATION\n")
                f.write("-"*30 + "\n")
                f.write(str(self.df.info()) + "\n\n")
                
                f.write("DESCRIPTIVE STATISTICS\n")
                f.write("-"*30 + "\n")
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    f.write(str(self.df[numeric_cols].describe()) + "\n\n")
                
                f.write("MISSING VALUES\n")
                f.write("-"*30 + "\n")
                missing = self.df.isnull().sum()
                for col, count in missing.items():
                    f.write(f"{col}: {count}\n")
            
            print(f"✅ Analysis report saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")


def main():
    """
    Main function to demonstrate comprehensive data analysis
    """
    print("🚀 ADVANCED DATA ANALYSIS WITH PANDAS & MATPLOTLIB")
    print("="*70)
    
    analyzer = DataAnalyzer()
    
    # Check if CSV dataset exists in the directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if csv_files:
        print(f"📁 Found CSV files: {csv_files}")
        # Try to load the first CSV file
        success = analyzer.load_dataset_from_csv(csv_files[0])
        if not success:
            print("📊 Using example dataset (Iris) instead...")
            analyzer.load_iris_dataset()
    else:
        print("📊 No CSV files found. Using example dataset (Iris)...")
        analyzer.load_iris_dataset()
    
    # Perform analysis tasks
    try:
        print("\n🔍 PERFORMING COMPREHENSIVE ANALYSIS")
        print("="*50)
        
        # Task 1: Dataset Exploration (already done above)
        
        # Task 2: Basic Statistical Analysis
        analyzer.basic_statistical_analysis()
        analyzer.groupby_analysis()
        
        # Task 3: Create Visualizations
        analyzer.create_visualizations()
        
        # Save analysis report
        analyzer.save_analysis_report()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📊 All visualizations have been displayed")
        print("📄 Analysis report has been saved")
        print("🎯 Dataset insights have been provided")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()
