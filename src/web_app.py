import dash
from dash import html, dcc, Input, Output, State, callback_context
import numpy as np
import joblib
import os
from datetime import datetime
import math

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server
app.title = "Wind Energy Weather Tool"

# Load models and scalers from the 'artifacts' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
model_1_path = os.path.join(current_dir, '..', 'artifacts', 'station1_model.pkl')
model_2_path = os.path.join(current_dir, '..', 'artifacts', 'station2_model.pkl')

scaler_1_path = os.path.join(current_dir, '..', 'artifacts', 'station1_scaler.pkl')
scaler_2_path = os.path.join(current_dir, '..', 'artifacts', 'station2_scaler.pkl')

# Load model and scaler data
model_1_data = joblib.load(model_1_path)
model_2_data = joblib.load(model_2_path)

scaler_1 = joblib.load(scaler_1_path)
scaler_2 = joblib.load(scaler_2_path)

model_1 = model_1_data['model']
model_2 = model_2_data['model']

# Energy production estimator (simplified version)
def estimate_energy_production(wind_speed):
    # Simplified logic, assumes wind speed to energy conversion
    estimated_energy = wind_speed * 0.1  # Placeholder conversion factor
    return f"Estimated Energy Production: {estimated_energy:.2f} kWh"

# Smart maintenance suggestor logic
def suggest_maintenance(wind_speed, snow_value, min_temp):
    maintenance_suggestions = []

    # Wind speed check for maintenance (e.g., if wind speed is too high)
    if wind_speed > 80:  # Example threshold for high wind speed
        maintenance_suggestions.append("Wind speed is high. Ensure turbines are inspected for possible wear.")

    # Snow accumulation check (e.g., if snow is significant)
    if snow_value > 10:  # Example threshold for snow accumulation (in inches)
        maintenance_suggestions.append("Heavy snow accumulation detected. Snow removal and turbine inspection recommended.")

    # Temperature check (e.g., if temperature is too low or too high)
    if min_temp < -10:  # Example threshold for low temperatures
        maintenance_suggestions.append("Extreme cold temperatures detected. Ensure turbines are operating within temperature limits.")
    elif min_temp > 30:  # Example threshold for high temperatures
        maintenance_suggestions.append("High temperatures detected. Ensure turbines are not overheating.")
    
    if not maintenance_suggestions:
        maintenance_suggestions.append("No maintenance needed at the moment.")

    return maintenance_suggestions

def maintenance_frequency(wind_speed, snow_value, min_temp):
    # Example conditions for maintenance frequency
    if wind_speed > 70 or snow_value > 10 or min_temp < -10:
        return "Maintenance Frequency: High"
    elif wind_speed > 40 or snow_value > 5:
        return "Maintenance Frequency: Medium"
    else:
        return "Maintenance Frequency: Low"


# Turbine efficiency score logic
def calculate_efficiency_score(wind_speed, snow_value, min_temp):
    # Set baseline values for optimal conditions
    ideal_wind_speed = 40  # Example ideal wind speed in km/h
    ideal_temp_range = (0, 25)  # Example ideal temperature range in °C
    ideal_snow_value = 0  # No snow is ideal for turbine performance

    # Calculate deviations from ideal conditions
    wind_speed_deviation = abs(wind_speed - ideal_wind_speed)  # Absolute difference
    temp_deviation = max(abs(min_temp - ideal_temp_range[0]), abs(min_temp - ideal_temp_range[1]))  # Deviations from ideal range
    snow_deviation = snow_value  # Snow negatively impacts efficiency

    # Combine all deviations into a score (simplified linear model)
    efficiency_score = 100 - (wind_speed_deviation * 0.5 + temp_deviation * 0.2 + snow_deviation * 0.3)

    # Ensure the score stays between 0 and 100
    efficiency_score = max(0, min(100, efficiency_score))

    return f"Turbine Efficiency Score: {efficiency_score:.2f}%"

# App layout
app.layout = html.Div([
    html.H1("Weather Forecaster", className='app-title'),

    html.Div([
        dcc.Dropdown(
            id='location-dropdown',
            options=[
                {'label': 'Raleigh-Durham International Airport — North Carolina, USA', 'value': 'location1'},
                {'label': 'Beach Corner, Edmonton — Alberta, Canada', 'value': 'location2'}
            ],
            placeholder="Select Location",
            className='dropdown'
        ),

        dcc.DatePickerSingle(
            id='date-picker',
            min_date_allowed=datetime(2000, 1, 1),
            max_date_allowed=datetime(2100, 1, 1),
            initial_visible_month=datetime.today(),
            date=datetime.today().date()
        ),

        html.Button("Predict", id='predict-btn', n_clicks=0, className='predict-btn'),
        html.Div(id='prediction-output', className='prediction-output'),
        html.Div(id='energy-output', className='energy-output'),  # For energy estimation
        html.Div(id='maintenance-output', className='maintenance-output'),  # For maintenance suggestion
        html.Div(id='efficiency-output', className='efficiency-output')  # For Turbine Efficiency Score
    ], className='form-container')
], className='main-container')


# Predict when button is clicked
@app.callback(
    [Output('prediction-output', 'children'),
     Output('energy-output', 'children'),
     Output('maintenance-output', 'children'),
     Output('efficiency-output', 'children')],  # Output for efficiency score
    Input('predict-btn', 'n_clicks'),
    Input('location-dropdown', 'value'),
    Input('date-picker', 'date')
)
def make_prediction(n_clicks, location_value, selected_date):
    ctx = callback_context
    if not ctx.triggered:
        return "", "", "", ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'location-dropdown' or trigger_id == 'date-picker':
        return "", "", "", ""

    if n_clicks == 0 or not location_value or not selected_date:
        return "", "", "", ""

    try:
        selected_date = datetime.strptime(selected_date, '%Y-%m-%d')
        month = selected_date.month
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
    except:
        return "Please select a valid date.", "", "", ""

    if location_value == 'location1':
        model = model_1
        scaler = scaler_1
        feature_names = [
            ("Max Temperature", "°C"),
            ("Min Temperature", "°C"),
            ("Average Temperature", "°C"),
            ("Average Wind Speed", "km/h"),
            ("Total Precipitation", "mm"),
            ("Average Dew Point", "°C"),
        ]
        snow_index = 6  # Make sure this index corresponds to Snow
        min_temp_index = 1
    else:
        model = model_2
        scaler = scaler_2
        feature_names = [
            ("Max Temperature", "°C"),
            ("Min Temperature", "°C"),
            ("Average Temperature", "°C"),
            ("Max Wind Speed", "km/h"),
        ]
        snow_index = 5
        min_temp_index = 1

    try:
        input_array = np.array([[month_sin, month_cos]])
        prediction = model.predict(input_array)

        if hasattr(scaler, 'inverse_transform'):
            prediction = scaler.inverse_transform(prediction.reshape(1, -1))

        # Snow override logic: If min_temp > 2, set snow to 0.0
        min_temp = prediction[0][min_temp_index]
        if min_temp > 2:
            prediction[0][snow_index] = 0.0  # Override snow to 0 if min_temp > 2

        # Collect predictions for output
        predicted_values = []
        wind_speed = float(prediction[0][3])  # Assuming wind speed is at index 3
        snow_value = prediction[0][snow_index]  # Get the snow value from the model

        # Append predicted values for features
        for i, (feature, unit) in enumerate(feature_names):
            value = float(prediction[0][i])
            predicted_values.append(html.P(f"Predicted {feature}: {value:.2f} {unit}"))

        # Explicitly append the snow prediction value
        predicted_values.append(html.P(f"Predicted Snow: {snow_value:.2f} in"))

        # Energy Estimator
        energy_estimate = estimate_energy_production(wind_speed)

        # Get maintenance suggestion
        maintenance_suggestions = suggest_maintenance(wind_speed, snow_value, min_temp)

        # Calculate Turbine Efficiency Score
        efficiency_score = calculate_efficiency_score(wind_speed, snow_value, min_temp)

        # Calculate Maintenance Frequency
        maintenance_freq = maintenance_frequency(wind_speed, snow_value, min_temp)


        # Combine maintenance suggestions and frequency into one output
        maintenance_output = html.Div([
            html.P("Maintenance Suggestions:"),
            html.Ul([html.Li(suggestion) for suggestion in maintenance_suggestions] + [html.Li(maintenance_freq)])
        ])

        # Return 4 outputs as expected
        return html.Div(predicted_values), energy_estimate, maintenance_output, efficiency_score

    except Exception as e:
        return html.Div([
            html.P("Error during prediction:"),
            html.Pre(str(e))
        ]), "", "", ""



# Run app
if __name__ == '__main__':
    app.run(debug=True)
