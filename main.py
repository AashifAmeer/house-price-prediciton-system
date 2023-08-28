import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "Housing.csv"  # Replace with the actual link or path

# Define lowercase column names
column_names = ["price", "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement",
                "hot_water_heating", "airconditioning", "parking", "prefarea", "furnishing_status"]

data = pd.read_csv(url, names=column_names, skiprows=1)  # Skip the first row

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=["mainroad", "guestroom", "basement", "hot_water_heating",
                                     "airconditioning", "prefarea", "furnishing_status"], drop_first=True)

# Split the dataset into features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Predict house prices
new_data = {
    "area": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "mainroad_yes": [1],
    "guestroom_yes": [0],
    "basement_yes": [1],
    "hot_water_heating_yes": [0],
    "airconditioning_yes": [1],
    "parking": [2],
    "prefarea_yes": [1],
    "furnishing_status_semi-furnished": [1],
    "furnishing_status_unfurnished": [0]
}

new_data_df = pd.DataFrame(new_data)
new_data_df = new_data_df[X_train.columns]

predicted_price = model.predict(new_data_df)
print("Predicted House Price:", predicted_price[0])
