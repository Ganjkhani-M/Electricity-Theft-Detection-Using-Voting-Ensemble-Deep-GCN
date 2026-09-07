# SGCC Electricity Theft Detection Dataset
The State Grid Corporation of China (SGCC) Electricity Theft Detection Dataset is a
real-world electricity consumption dataset released by the State Grid Corporation of
China. It is one of the most widely used benchmark datasets in the field of electricity 
theft detection and non-technical loss (NTL) identification in smart grids. 


# Dataset Overview
The dataset contains electricity consumption records from 42,372 unique consumers over a
period of 1,035 consecutive days, spanning from January 1, 2014, to October 30, 2016.
The data is provided in a single CSV file with 1,037 columns, and 42,372 rows. 


# Features 
1: CONS_NO: Consumer Number stands for a customer ID of string type. 
2: 'MM/DD/YYYY': The electric consumption on a given day. 
3: FLAG: 0 indicating no theft and 1 for theft. 


# Class Distribution
The dataset is highly imbalanced, which makes it particularly suitable for research on class
imbalance handling techniques:

Class Type: Normal (FLAG = 0) / Count = 38,757 / Percentage = ~91.46%
Class Type: Theft  (FLAG = 1) / Count = 3,615  / Percentage = ~8.54


# Notes on Data Quality
1: The dataset contains missing values that require preprocessing.
2: Dates are formatted as "MM/DD/YYYY'. 
3: The data requires light cleaning before use in machine learning models. 


# Why This Dataset?
The SGCC dataset is a real-world, large-scale electricity consumption dataset that has 
been extensively used in academic research on electricity theft detection. Its realistic 
class imbalance, temporal span, and large consumer base make it an ideal benchmark for 
evaluating the performance of theft detection algorithms in smart grid environments. 


# How to Download the Dataset?
The dataset is available on this link: https://drive.google.com/file/d/1wy9YGnJn9LowRKe8ALvQvHaRL8HLAJTF/view?usp=sharing

