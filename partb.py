import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from google.colab import drive
drive.mount('/content/drive')


# Load the training data and split into features and target
# Drop the 'marker' column from the features
# The 'marker' column is the target variable
# The marker coloumn is the last coloumn 129th coloumn however the better strategy is locating the column by name
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TrainingDataBinary.csv')
X = data.drop('marker', axis=1)
# The marker coloumn is the last coloumn 129th coloumn however the better strategy is locating the column by name
y = data['marker']


# Normalize the features using MinMaxScaler
# the MinMaxScaler scales the data to a fixed range [0, 1]
# this is done to ensure that the model is not biased towards features with larger magnitude so it can train on the scaled data
# The min-max scaler is good for models that are sensitive to the magnitude of the features there are 128 parameters
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
# the training set is 85% of the data and the validation set is 15% of the data
# the random state is set to 42 so that the split is reproducible
# the random state is the seed used by the random number generator so that the split is the same each time
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Initialise and configure the XGBoost model with the specified hyperparameters
# We did hyperparameter tuning to get the best hyperparameters for the model using the f1 score
# The learning rate is the step size at each iteration while moving towards a minimum of a loss function
# The max depth is the maximum depth of the tree which helps to control overfitting
# The n_estimators is the number of trees and the more there are the better the model
# eval_metric being logless is the loss function of the model which tracks the error rate
xgb_model = XGBClassifier(learning_rate=0.2, max_depth=5, n_estimators=300, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Evaluate the model on the validation set
# Predict on the validation set and evaluate
# The accuracy is the number of correct predictions made by the model divided by the total number of predictions
# The precision is the number of true positive predictions divided by the number of true positive and false positive predictions
# The F1-score is worked out by the harmonic mean of the precision and recall (2 * (precision * recall) / (precision + recall)))
# The confusion matrix is a table that is used to describe the performance of a classification model it shows how many true positives, true negat
y_pred = xgb_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')
confusion = confusion_matrix(y_val, y_pred)

metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'F1 Score': f1,
    'Confusion Matrix': confusion
}

for metric_name, metric_value in metrics.items():
    print(f'{metric_name}: {metric_value}')


# Load the test data which is the data which doesn't have the target variable 'marker' and we are using the training data to predict it
# The test data is loaded and normalized using the MinMaxScaler
test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/TestingDataMulti.csv')
X_test = scaler.transform(test_data)  # Normalize test data

# Predict the 'marker' column for the test data using the trained model
predicted_markers = xgb_model.predict(X_test)

# Convert predictions to float to maintain the desired numeric format as marker coloumn is  0.0 or 1.0 to keep the accuracy the same
predicted_markers = predicted_markers.astype(float)

# Add the predictions as a new column in the test data DataFrame
test_data['marker'] = predicted_markers

# Save the DataFrame with the predictions to a new CSV file and print statement to confirm the saving of the file
test_data.to_csv('/content/drive/My Drive/Colab Notebooks/TestingResultsBestModel_B.csv', index=False)
print("The predictions have been saved successfully to 'TestingResultsBestModel_B.csv'.")
