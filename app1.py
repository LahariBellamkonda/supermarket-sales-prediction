from flask import Flask, render_template, request
import pandas as pd
import joblib
import math

# Initialize Flask app
app = Flask(__name__)

# Load the trained Random Forest model
random_forest_model = joblib.load('random_forest_model.pkl')

# Function to preprocess input data and make predictions
def predict_sales(input_data):
    # Load the trained encoder
    encoder = joblib.load('encoder.pkl')

    # Define the order of columns for one-hot encoding
    one_hot_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    # Perform one-hot encoding on the user input
    input_data_encoded = pd.get_dummies(input_data, columns=one_hot_cols)

    # Ensure that the one-hot encoded input data contains all the necessary columns
    df_train_encoded = joblib.load('df_train_encoded.pkl')
    for col in df_train_encoded.columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match the model's training data
    input_data_encoded = input_data_encoded[df_train_encoded.columns]

    # Make a prediction using the Random Forest model
    predicted_sales = random_forest_model.predict(input_data_encoded)
    # Calculate the number of items predicted
    item_mrp = input_data['Item_MRP'].values[0]
    predicted_items_float = predicted_sales / item_mrp
    predicted_items = math.ceil(predicted_items_float)  # Round up to the nearest integer

    return predicted_sales, predicted_items

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the HTML form
    item_weight = float(request.form['Item_Weight'])
    item_fat_content = request.form['Item_Fat_Content']
    item_visibility = float(request.form['Item_Visibility'])
    item_type = request.form['Item_Type']
    item_mrp = float(request.form['Item_MRP'])
    outlet_identifier = request.form['Outlet_Identifier']
    outlet_establishment_year = int(request.form['Outlet_Establishment_Year'])
    outlet_size = request.form['Outlet_Size']
    outlet_location_type = request.form['Outlet_Location_Type']
    outlet_type = request.form['Outlet_Type']

    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Identifier': [outlet_identifier],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type]
    })

    # Make a prediction using the input data
    predicted_sales, predicted_items = predict_sales(input_data)

    return render_template('result.html', predicted_sales=predicted_sales, predicted_items=predicted_items)

if __name__ == '__main__':
    app.run(debug=True)
