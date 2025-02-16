# Load necessary libraries
library(ggplot2)
library(dplyr)
library(reshape2)
library(tidyr)

# Read the dataset
data <- read.csv("diabetes.csv")

# Function to count missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))

# Function to detect outliers using the IQR method
count_outliers <- function(x) {
  if (is.numeric(x)) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
  } else {
    return(NA)
  }
}
outliers_count <- sapply(data, count_outliers)

# Function to check skewness
check_skewness <- function(x) {
  if (is.numeric(x)) {
    skew_value <- e1071::skewness(x, na.rm = TRUE)
    if (abs(skew_value) > 1) {
      return("High Skew")
    } else if (abs(skew_value) > 0.5) {
      return("Moderate Skew")
    } else {
      return("Low Skew")
    }
  } else {
    return(NA)
  }
}
skewness_status <- sapply(data, check_skewness)

# Identify factors (categorical features)
factors_status <- sapply(data, function(x) if (is.factor(x)) "Factor" else "Not Factor")

# Create a summary table
summary_table <- data.frame(
  Feature = names(data),
  MissingValues = missing_values,
  OutliersCount = outliers_count,
  Skewness = skewness_status,
  IsFactor = factors_status
)

# Display the table
print(summary_table)

# Handling missing values: Impute with median
data$SkinThickness[is.na(data$SkinThickness)] <- median(data$SkinThickness, na.rm = TRUE)
data$Insulin[is.na(data$Insulin)] <- median(data$Insulin, na.rm = TRUE)
data$BloodPressure[is.na(data$BloodPressure)] <- median(data$BloodPressure, na.rm = TRUE)

# Histogram of Glucose Levels
ggplot(data, aes(x = Glucose)) + 
  geom_histogram(binwidth = 10, fill = "skyblue", color = "black") + 
  labs(title = "Distribution of Glucose Levels", x = "Glucose", y = "Frequency")

# Correlation Heatmap
cor_matrix <- cor(data %>% select(-Outcome))
melted_cor <- melt(cor_matrix)
ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap", x = "Features", y = "Features")

# Box Plot of Insulin Levels by Outcome
ggplot(data, aes(x = factor(Outcome), y = Insulin, fill = factor(Outcome))) + 
  geom_boxplot() +
  labs(title = "Box Plot of Insulin Levels by Outcome", x = "Outcome", y = "Insulin Levels")

# Scatter Plot of BMI vs. Age colored by Outcome
ggplot(data, aes(x = Age, y = BMI, color = factor(Outcome))) + 
  geom_point(alpha = 0.6) +
  labs(title = "Scatter Plot of BMI vs. Age", x = "Age", y = "BMI")

# Count Plot of Pregnancies by Outcome
ggplot(data, aes(x = factor(Pregnancies), fill = factor(Outcome))) + 
  geom_bar(position = "dodge") +
  labs(title = "Count Plot of Pregnancies by Outcome", x = "Number of Pregnancies", y = "Count")
