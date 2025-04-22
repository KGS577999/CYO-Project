# Wind Energy Weather Tool

A web application powered by machine learning that predicts weather-related variables such as wind speed, temperature, and snow accumulation, and estimates wind turbine energy production based on these forecasts. The application also provides maintenance suggestions and operational insights for wind farms based on weather conditions.

## Features

- **Predicts weather variables**: Wind speed, temperature, snow accumulation, etc.
- **Estimates energy production**: Based on weather data and wind turbine characteristics.
- **Maintenance suggestions**: Based on predicted weather conditions like temperature and wind speed.
- **Real-time analysis**: Interactive Dash web app for live predictions and insights.
- **Hybrid model**: Combines machine learning with rule-based logic for more reliable predictions.

## How It Works

The application uses a trained machine learning model to predict key weather variables and estimate wind energy production. The model is based on historical weather data and includes both raw features (such as temperature and wind speed) and engineered features. Additionally, it provides maintenance recommendations based on specific weather thresholds.

Key model features include:
- Wind speed
- Temperature
- Snow accumulation
- Precipitation
- Maintenance suggestions (based on weather conditions)
  
The model was trained on historical weather data for the relevant locations.

## Technologies Used

- **Python** – Core programming language
- **Dash (Plotly)** – Web application framework for real-time analysis
- **Scikit-learn** – For machine learning model training & preprocessing
- **Gunicorn** – For production-ready web server
- **NumPy / Pandas** – Data manipulation and processing
- **Joblib** – For model serialization
- **Render** – Cloud hosting platform

## Project Structure

├── artifacts/
│   ├── stationl_model.pkl
│   ├── station2_model.pkl
│   ├── stationl_scaler.pkl
│   ├── station2_scaler.pkl
├── data/
    ├── USA.cv
│   └── Canada.csv
├── notebooks/
│   └── model_training.ipynb    # Full model training and evaluation notebook
├── src/
│   ├── assets/
│   │   └── styles.css          # Custom CSS styles for the web app
│   └── web_app.py              # Main Dash web application
├── .gitignore
├── Procfile                    # Instructions for Render deployment
├── ReadMe.md
└── requirements.txt            # Python dependencies for the project


## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/KGS577999/CYO-Project
cd CYO-Project
```

### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
cd src
python web_app.py
```

Open your browser and visit http://localhost:8050 to interact with the app.

## Model Details

- **Model Type**: Random Forest, Linear Regression, or XGBoost (based on experimentation)
- **Input Features**: Date and location.
- **Output**: Weather prediction (wind speed, temperature, precipitation, snowfall) and energy production estimate
- **Data Preprocessing**: Standard scaling, feature engineering for cyclical dates
- **Model Serialization**: Saved using joblib for production deployment

## Output Fields examples (UI)

| Feature | Type | Example |
|---------|------|---------|
| Wind Speed | Numeric | 12 m/s |
| Temperature | Numeric | 15°C |
| Snow Accumulation | Numeric | 2 cm |
| Maintenance Flag | Binary | Yes/No |

## Risk Assessment Logic

The app calculates a Maintenance Score based on key weather inputs:

```
maintenance_score = 0.4 * wind_speed + 0.3 * temperature + 0.3 * snow_accumulation
```

Risk Categories:
- **Low Risk**: Score ≥ 0.7
- **Moderate Risk**: 0.4 ≤ Score < 0.7
- **High Risk**: Score < 0.4

## Recommendations System

Based on the model output and input conditions, the app provides maintenance suggestions, such as:
- Schedule maintenance for turbines exposed to high wind speeds or extreme temperatures
- Clean snow buildup on turbines if snow accumulation exceeds a threshold
- Adjust operational settings based on predicted low energy production periods

These suggestions are provided automatically after each prediction.

## Deployment Notes

To deploy on platforms like Render:
- Add a Procfile:
  ```
  web: gunicorn src.web_app:server
  ```
- Ensure the application dynamically handles the port:
  ```python
  port = int(os.environ.get("PORT", 8050))
  app.run(host="0.0.0.0", port=port)
  ```
- Ensure that Render (or another platform) is set up to install dependencies from requirements.txt and run gunicorn.

## Contributors

This project was collaboratively developed by:

- M.C. Wolmarans
- Caitlin Burnett
- Kyle Smith
- Paul-Dieter Brandt

## Future Improvements

- Integrate real-time weather data from external APIs
- Implement user-specific settings for wind turbine operations
- Expand model features to include wind direction, humidity, etc.
- Improve energy prediction accuracy with additional data sources
- Provide more detailed maintenance reports and notifications

