from src.mlproject.pipeline.predictionpipeline import PredictionPipeline
import pandas as pd

# Sample input data
sample_data = {
    'Airline': ['IndiGo'],
    'Source': ['CXB'],
    'Destination': ['CCU'],
    'Stopovers': ['Direct'],
    'Class': ['Business'],
    'Booking Source': ['Online Website'],
    'Days Before Departure': [10],
    'Arrival Time': ['Morning'],
    'Departure Time': ['Morning']
}
input_df = pd.DataFrame(sample_data)

# Initialize the PredictionPipeline
pipeline = PredictionPipeline()

try:
    # Make a prediction
    prediction = pipeline.predict(input_df)
    print("\n=== PREDICTION RESULT ===")
    print("Prediction:", prediction)
except Exception as e:
    print("Error during prediction:", str(e))
    print('')