import dash
from dash import html, dcc, Input, Output, State, callback_context
import numpy as np
import joblib
import os

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Load models and scalers
current_dir = os.path.dirname(os.path.abspath(__file__))
model_1_path = os.path.join(current_dir, '..', 'artifacts', 'station1_model.pkl')
model_2_path = os.path.join(current_dir, '..', 'artifacts', 'station2_model.pkl')

model_1_data = joblib.load(model_1_path)
model_2_data = joblib.load(model_2_path)

model_1 = model_1_data['model']
scaler_1 = model_1_data['scaler']
feature_names_1 = model_1_data['feature_names']

model_2 = model_2_data['model']
scaler_2 = model_2_data['scaler']
feature_names_2 = model_2_data['feature_names']

location_features = {
    'location1': ['Max Temperature', 'Min Temperature', 'Avg Dew Point', 'Avg Relative Humidity', 'Temp Range', 'Dew Deficit', 'Heat Index', 'Precip Flag', 'Dry Streak', 'Temp Pca1'],
    'location2': ['Days Precip_ge_0.01in', 'Max Daily Precip', 'Max Temperature', 'Min Temperature', 'Wind Direction Peak', 'Wind Speed Peak', 'Temp Range', 'Precip Flag', 'Dry Streak', 'Precip Pca1']
}

location1_features = location_features['location1']
location2_features = location_features['location2']

# App layout
app.layout = html.Div([
    html.H1("Weather Forecast Predictor", className='app-title'),

    html.Div([
        dcc.Dropdown(
            id='location-dropdown',
            options=[
                {'label': 'Location 1', 'value': 'location1'},
                {'label': 'Location 2', 'value': 'location2'}
            ],
            placeholder="Select Location",
            className='dropdown'
        ),

        html.Div(id='input-container', className='form-group'),

        html.Button("Predict", id='predict-btn', n_clicks=0, className='predict-btn'),
        html.Div(id='prediction-output', className='prediction-output')
    ], className='form-container')
], className='main-container')

# Update input fields based on selected location
@app.callback(
    Output('input-container', 'children'),
    Input('location-dropdown', 'value')
)
def update_input_fields(location_value):
    if not location_value:
        return []

    features = location1_features if location_value == 'location1' else location2_features

    inputs = []
    for feature in features:
        inputs.append(html.Div([
            html.Label(f"{feature}:", style={'font-weight': 'bold'}),
            dcc.Input(
                id={'type': 'feature-input', 'index': feature},
                type='number',
                placeholder=f"Enter {feature}",
                debounce=True,
                style={'margin-bottom': '10px', 'width': '50%'}
            )
        ]))
    return inputs

# Predict when button is clicked
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('location-dropdown', 'value'),
    State({'type': 'feature-input', 'index': dash.ALL}, 'value'),
    State({'type': 'feature-input', 'index': dash.ALL}, 'id')
)
def make_prediction(n_clicks, location_value, input_values, input_ids):
    ctx = callback_context
    if not ctx.triggered:
        return ""

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # If location dropdown was changed, clear prediction output
    if trigger_id == 'location-dropdown':
        return ""

    if n_clicks == 0 or not location_value:
        return ""

    # Match inputs to feature names
    input_dict = {item['index']: val for item, val in zip(input_ids, input_values)}

    if location_value == 'location1':
        selected_features = location1_features
        model = model_1
        scaler = scaler_1
    else:
        selected_features = location2_features
        model = model_2
        scaler = scaler_2

    try:
        input_array = np.array([input_dict[feat] for feat in selected_features]).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        if hasattr(prediction, '__len__') and len(prediction[0]) == 2:
            return html.Div([
                html.P(f"Predicted Avg Temperature: {prediction[0][0]:.2f}"),
                html.P(f"Predicted Total Precipitation: {prediction[0][1]:.2f}")
            ])
        else:
            return html.P(f"Prediction: {prediction[0]:.2f}")

    except Exception as e:
        return html.Div([
            html.P("Error during prediction:"),
            html.Pre(str(e))
        ])


# Run app
if __name__ == '__main__':
    app.run(debug=True)
