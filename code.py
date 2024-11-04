# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load data from CSV
df = pd.read_csv("Breast_cancer_data.csv")  # Replace with the path to your CSV file

# Display the first few rows of the data
print("First five rows of the dataset:")
print(df.head())

# Explore basic statistics of the dataset
print("\nDataset summary statistics:")
print(df.describe())

# Check for any missing values
print("\nChecking for missing values in the dataset:")
print(df.isnull().sum())

# Explore data distribution for the target variable
print("\nDiagnosis class distribution:")
print(df['diagnosis'].value_counts())

# Visualize the distribution of classes
sns.countplot(data=df, x='diagnosis')
plt.title("Diagnosis Class Distribution")
plt.show()

# Assuming 'diagnosis' is the target column
X = df.drop(columns=["diagnosis"])  # Extract features
y = df["diagnosis"]  # Extract target

# Encode target labels if necessary
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nDataset split into train and test sets.")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling applied to the dataset.")

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
print("\nSVM classifier initialized with linear kernel.")

# Perform cross-validation to assess model stability
print("\nPerforming cross-validation:")
cv_scores = cross_val_score(svm_classifier, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", np.mean(cv_scores))

# Train SVM classifier
svm_classifier.fit(X_train_scaled, y_train)
print("\nSVM classifier training completed.")

# Predict on test set
y_pred = svm_classifier.predict(X_test_scaled)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy on test set:", accuracy)

# Display classification report for additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Perform hyperparameter tuning using GridSearchCV
print("\nPerforming hyperparameter tuning using GridSearchCV:")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("\nBest hyperparameters found:", best_params)
print("Best cross-validation score from grid search:", grid_search.best_score_)

# Retrain the model with best hyperparameters
print("\nRetraining the SVM classifier with optimized hyperparameters...")
best_estimator.fit(X_train_scaled, y_train)
y_pred_optimized = best_estimator.predict(X_test_scaled)

# Evaluate optimized model
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
print("\nOptimized model accuracy on test set:", optimized_accuracy)

# Visualize decision boundaries using PCA for dimensionality reduction (if applicable)
print("\nPerforming PCA for visualization of decision boundaries (2D):")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
best_estimator.fit(X_train_pca, y_train)

# Plot decision boundaries
plt.figure(figsize=(10, 6))
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = best_estimator.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=20)
plt.legend(handles=scatter.legend_elements()[0], labels=label_encoder.classes_)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("SVM Decision Boundaries (2D PCA)")
plt.show()
