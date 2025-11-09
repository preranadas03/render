from flask import Flask, request, render_template
import numpy as np
import pickle

# -------------------------
# Load model and encoder
# -------------------------
model_path = 'model.pkl'
encoder_path = 'encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        ship_type = request.form['ship_type']
        technical_eff = float(request.form['technical_efficiency'])
        fuel_cons = float(request.form['fuel_consumption'])
        co2_emissions = float(request.form['co2_emissions'])
        time_at_sea_annual = float(request.form['annual_time_sea'])
        avg_fuel_dist = float(request.form['avg_fuel_distance'])
        avg_fuel_work = float(request.form['avg_fuel_work'])
        avg_co2_work = float(request.form['avg_co2_work'])
        time_spent_sea = float(request.form['time_at_sea'])

        # Encode ship type
        ship_type_encoded = encoder.transform([ship_type])[0]

        # Prepare final input
        input_features = np.array([
            [
                technical_eff,
                fuel_cons,
                co2_emissions,
                time_at_sea_annual,
                avg_fuel_dist,
                avg_fuel_work,
                avg_co2_work,
                time_spent_sea,
                ship_type_encoded
            ]
        ])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Interpret the prediction
        # (You can tweak the threshold based on your dataset)
        if prediction < 50:
            result_text = f"🌿 Low Emission — Predicted CO₂ Output: {prediction:.2f}"
        elif prediction < 150:
            result_text = f"⚖️ Moderate Emission — Predicted CO₂ Output: {prediction:.2f}"
        else:
            result_text = f"🔥 High Emission — Predicted CO₂ Output: {prediction:.2f}"

        return render_template('index.html', prediction_text=result_text)

    except Exception:
        return render_template(
            'index.html',
            prediction_text="⚠️ Something went wrong. Please check your input values and try again."
        )



if __name__ == "__main__":
    app.run(debug=True)
