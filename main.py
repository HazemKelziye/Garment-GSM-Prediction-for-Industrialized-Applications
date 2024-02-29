import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from convert_func import convert_fraction_to_float
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

csv_file_path = "/dataset/updated_garment_dataset2.csv"
df = pd.read_csv(csv_file_path)

# Dropping unnecessary features
df = df.drop(['sample id', 'machine id', 'total input yarn[kg]', 'fabric weight[kg]',
              'total revolutions per roll ', 'fabric roll weight', 'yarn 1 make', 'yarn 2 make',
              'yarn 3 make', 'machine speed[rpm]', 'roll width[cm]'], axis=1)

# Apply the conversion function
# iplik numarasi stands for yarn count
columns_to_convert = ['iplik 1 numarasi[x m/1gr]', 'iplik 2 numarasi[x m/1gr]', 'iplik 3 numarasi[x m/1gr]']
for column in columns_to_convert:
    df[column] = df[column].apply(convert_fraction_to_float).fillna(0.0000).astype(float)

# Handle missing values for other numerical columns
numerical_columns = [0, 1, 4, 6, 7, 8, 10, 11, 12, 14, 15, 18]  # Adjust based on your DataFrame structure
df.iloc[:, numerical_columns] = df.iloc[:, numerical_columns].fillna(0)

cat_features = ['machine type', 'fabric type', 'yarn 1 type',
                'yarn 2 type', 'yarn 3 type', 'machine caliber', 'fabric property[roll?]']

# Split the data into training and test sets before encoding
X = df.drop(['GSM[gr/m2]'], axis=1)
y = df['GSM[gr/m2]']  # Specify the target column based on your dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Encode categorical variables using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_features_train = encoder.fit_transform(X_train[cat_features])
encoded_features_test = encoder.transform(X_test[cat_features])

# Create DataFrames from the encoded features and rename columns to match original features
encoded_df_train = pd.DataFrame(encoded_features_train, columns=encoder.get_feature_names_out(cat_features),
                                index=X_train.index)
encoded_df_test = pd.DataFrame(encoded_features_test, columns=encoder.get_feature_names_out(cat_features),
                               index=X_test.index)

# Concatenate the encoded features back with the original dataset
X_train = pd.concat([X_train.drop(cat_features, axis=1), encoded_df_train], axis=1)
X_test = pd.concat([X_test.drop(cat_features, axis=1), encoded_df_test], axis=1)


################################ Training ################################
# Training part with RandomForest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust the n_estimators and random_state as needed
rf_reg.fit(X_train, y_train)
garment_predictions = rf_reg.predict(X_train)
rf_mse = mean_squared_error(y_train, garment_predictions)
rf_rmse = np.sqrt(rf_mse)

# Using cross-validation to evaluate model performance
scores = cross_val_score(rf_reg, X_train, y_train, scoring="neg_mean_squared_error",
                         cv=100)  # Changed from 100 to 10 for performance, adjust as needed
rf_rmse_scores = np.sqrt(-scores)

# Print out the cross-validation results
print(rf_rmse_scores)
print(rf_rmse_scores.mean())

# Testing part
test_predictions = rf_reg.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)

# Print out the results
print(f"Test RMSE: {test_rmse}")
joblib.dump(rf_reg, "trained_models/random_forests/finalized_random_forest_reg_model_1.sav")
