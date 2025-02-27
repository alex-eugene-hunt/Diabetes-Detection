# Early Detection of Diabetes Using Machine Learning Classification Algorithms

## Project Overview
This project implements and compares various machine learning classification algorithms for the early detection of diabetes. The system uses both original and synthetic datasets to train and evaluate multiple models, providing comprehensive insights into their performance and reliability.

## Technical Stack
- **Programming Language:** R
- **IDE:** R Studio
- **Version Control:** Git

### Key Libraries
- `caret`: For machine learning model training and evaluation
- `randomForest`: Implementation of Random Forest algorithm
- `e1071`: Support Vector Machine implementation
- `gbm`: Gradient Boosting Machine implementation
- `ggplot2`: Data visualization
- `reshape2`: Data transformation
- `pROC`: ROC curve analysis
- `tidyr` & `dplyr`: Data manipulation and cleaning

## Algorithms Implemented

![Screenshot 2025-02-27 140934](https://github.com/user-attachments/assets/9544822e-c3d0-4bf3-a7b9-87f7490d753d)


1. **Logistic Regression**
   - Binary classification model
   - Probability-based predictions
   - Baseline model for comparison

2. **Random Forest**
   - Ensemble learning method
   - 100 decision trees
   - Feature importance analysis

3. **Support Vector Machine (SVM)**
   - Linear kernel
   - Probability estimates enabled
   - Hyperplane-based classification

4. **Gradient Boosting Machine (GBM)**
   - Bernoulli distribution
   - 100 trees
   - Interaction depth of 3

## Data Processing Pipeline
1. **Data Loading**
   - Original dataset from CSV
   - Synthetic dataset from RData file

![Screenshot 2025-02-27 141002](https://github.com/user-attachments/assets/53303fca-4c30-4120-89f9-f9eeb7f0a03a)


2. **Preprocessing**
   - Missing value handling for key metrics:
     - Glucose
     - Blood Pressure
     - Skin Thickness
     - Insulin
     - BMI
   - Zero value replacement with NA
   - Data normalization

3. **Data Splitting**
   - 70/30 train-test split
   - Stratified sampling using `createDataPartition`
   - Seed setting for reproducibility

## Model Evaluation Metrics
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curves
- AUC (Area Under Curve)

![Screenshot 2025-02-27 140907](https://github.com/user-attachments/assets/c9e71fbe-6660-404d-9ddb-2b247c3d05c4)


## Visualization Techniques
- Performance comparison plots
- Feature importance visualization
- ROC curve comparisons
- Metric distribution analysis

## Key Features
- **Dual Dataset Analysis:** Comparison between original and synthetic data
- **Cross-Validation:** Robust model evaluation
- **Ensemble Methods:** Combination of multiple algorithms
- **Performance Metrics:** Comprehensive evaluation framework
- **Reproducible Research:** Seed setting and documented workflow

## Project Structure
```
├── R Scripts/
│   ├── Classification_Diabetes_Algorithms.R  # Main classification algorithms
│   ├── Data_Preprocessing_Analyzation.R      # Data preprocessing
│   └── InitialDataAnalysis.R                # Initial data exploration
├── Data/
│   ├── Diabetes_Dataset.csv                 # Original dataset
│   └── synthetic_density_data.RData         # Synthetic dataset
└── README.md
```

## Model Performance
The project evaluates each model's performance using both original and synthetic datasets, providing insights into:
- Model accuracy and reliability
- Overfitting detection
- Generalization capabilities
- Performance stability

## Future Improvements
- Hyperparameter optimization
- Deep learning implementation
- Feature engineering expansion
- Real-time prediction capabilities

## Requirements
- R version 3.6.0 or higher
- Required R packages listed in the Technical Stack section
- Minimum 8GB RAM recommended for optimal performance

## Reproduction Steps
1. Clone the repository
2. Install required R packages
3. Run the preprocessing script
4. Execute the classification algorithm script
5. Analyze results through visualization outputs

## Contributors
- Alex Hunt
- Oluchi Ejehu
- Zainab Iyiola

## Course Information
DSA-5103-995: Final Project FA 2024
Professor Nicholson
