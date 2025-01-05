# Exploratory Data Analysis

[![Build Status](https://github.com/notGiGi/EDA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/notGiGi/EDA.jl/actions/workflows/CI.yml?query=branch%3Amain)


**EDA.jl** is a powerful and comprehensive Julia package designed to streamline exploratory data analysis (EDA). This package equips data analysts, scientists, and researchers with tools to effortlessly clean, analyze, and visualize data, enabling insightful decision-making and robust analyses.

## Key Features

### Data Cleaning and Handling
- Identify and handle missing data with customizable thresholds.
- Detect and manage outliers using statistical methods like Interquartile Range (IQR).
- Remove columns based on correlation or missing data percentage.

### Data Exploration
- Generate detailed previews of datasets with `visualize_data`.
- Examine data types and detect anomalies in column formats with `dataType`.
- Compute and visualize correlation matrices to understand relationships between variables.

### Visualization
- Create heatmaps of correlation matrices for intuitive data exploration.
- Generate interactive visualizations of correlation networks, highlighting variable relationships.
- Plot real vs. predicted values in regression models for deeper model evaluation.

### Statistical Analysis
- Compute and display correlation matrices for numerical columns.
- Fit and evaluate linear regression models with detailed coefficient summaries.

### Robust Structure
- Encapsulates operations within the `EDALoader` structure for streamlined data management.
- Includes built-in caching to improve performance on repeated operations.

---

## Why Choose EDA.jl?

EDA.jl simplifies the data analysis workflow, enabling users to focus on extracting insights rather than grappling with tedious preprocessing. Its intuitive functions and comprehensive toolkit make it the ideal companion for data exploration tasks in Julia.

Whether you are working on small-scale projects or handling large datasets, EDA.jl adapts to your needs with its modular design and high-performance capabilities.

---

## Examples of Usage

### Load and Explore Data

```julia
using EDA
using DataFrames

# Sample data
data = DataFrame(
    A = [1, 2, 3, missing, 5],
    B = [2, 4, 6, 8, missing],
    C = [5, 4, 3, 2, 1]
)

# Initialize the EDA loader
loader = EDALoader(data)

# Preview the first few rows
println(visualize_data(loader, n=3))

# Examine column data types
println(dataType(loader))

# Remove columns with more than 50% missing data
cleaned_data = threshold(loader, 50.0)
println(cleaned_data)

# Generate a correlation matrix
println(correlation(loader))

# Visualize correlations as a heatmap
heat(loader)

# Perform linear regression
linearregression(loader, "B", ["A", "C"])

# Create a correlation network graph
correlation_network(loader, threshold=0.5)

# Create a correlation network graph
correlation_network(loader, threshold=0.5)

```

- **Planned Enhancements**:
    -Time-Series Analysis: Tools for decomposing, forecasting, and detecting anomalies in time-series data.
    -Interactive Dashboards: Integration with interactive visualization libraries like PlotlyJS.
    -Machine Learning Integration: Seamless compatibility with Julia's ML ecosystem.
