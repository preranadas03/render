from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import os

# -------------------------
# Load model and encoder
# -------------------------
model_path = 'model.pkl'
encoder_path = 'encoder.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

# The machine learning model is trained on these 4 exact categories.
# Selecting other categories will cause the encoder to fail.
VALID_SHIP_TYPES = ['Bulk carrier', 'General cargo ship', 'Refrigerated cargo carrier', 'Ro-ro ship']

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
        ship_type = request.form.get('ship_type')
        if ship_type not in VALID_SHIP_TYPES:
            return render_template(
                'index.html',
                prediction_text=f"⚠️ Unsupported ship type: '{ship_type}'. Please choose from: {', '.join(VALID_SHIP_TYPES)}."
            )

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

        # Prepare final input (expects exactly 5 features)
        input_features = np.array([
            [
                technical_eff,
                co2_emissions,
                avg_fuel_dist,
                time_spent_sea,
                ship_type_encoded
            ]
        ])

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Interpret the prediction
        if prediction < 50:
            result_text = f"🌿 Low Emission — Predicted CO₂ Output: {prediction:.2f}"
        elif prediction < 150:
            result_text = f"⚖️ Moderate Emission — Predicted CO₂ Output: {prediction:.2f}"
        else:
            result_text = f"🔥 High Emission — Predicted CO₂ Output: {prediction:.2f}"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"⚠️ Something went wrong: {str(e)}. Please check your input values and try again."
        )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Check if JSON data or form data is provided
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract values
        ship_type = data.get('ship_type')
        if ship_type not in VALID_SHIP_TYPES:
            return jsonify({
                "status": "error",
                "message": f"Unsupported ship type '{ship_type}'. Supported: {VALID_SHIP_TYPES}"
            }), 400

        technical_eff = float(data['technical_efficiency'])
        fuel_cons = float(data['fuel_consumption'])
        co2_emissions = float(data['co2_emissions'])
        time_at_sea_annual = float(data['annual_time_sea'])
        avg_fuel_dist = float(data['avg_fuel_distance'])
        avg_fuel_work = float(data['avg_fuel_work'])
        avg_co2_work = float(data['avg_co2_work'])
        time_spent_sea = float(data['time_at_sea'])

        # Encode ship type
        ship_type_encoded = encoder.transform([ship_type])[0]

        # Prepare final input (expects exactly 5 features)
        input_features = np.array([
            [
                technical_eff,
                co2_emissions,
                avg_fuel_dist,
                time_spent_sea,
                ship_type_encoded
            ]
        ])

        # Make prediction
        prediction = float(model.predict(input_features)[0])

        # Interpret the prediction and assign class & ratings
        if prediction < 50:
            category = "Low"
            status_text = "🌿 Low Emission"
            color_class = "prediction-low"
            color_hex = "#2ecc71"
            cii_rating = "A" if prediction < 30 else "B"
        elif prediction < 150:
            category = "Moderate"
            status_text = "⚖️ Moderate Emission"
            color_class = "prediction-moderate"
            color_hex = "#f1c40f"
            cii_rating = "C" if prediction < 100 else "D"
        else:
            category = "High"
            status_text = "🔥 High Emission"
            color_class = "prediction-high"
            color_hex = "#e74c3c"
            cii_rating = "E"

        # Eco-equivalents calculations (based on predicted carbon index rate and typical shipping profile)
        # Assumed standard operating distance: 50,000 nautical miles per year.
        annual_est_emissions_tonnes = (prediction * 50000) / 1000000.0  # tonnes of CO2
        # 1 tree absorbs ~22kg = 0.022 tonnes of CO2 per year
        trees_offset = int(annual_est_emissions_tonnes / 0.022)
        # Average car emits ~4.6 tonnes of CO2 per year
        cars_offset = round(annual_est_emissions_tonnes / 4.6, 1)
        # Average home electricity emits ~4 tonnes of CO2 per year
        homes_offset = round(annual_est_emissions_tonnes / 4.0, 1)

        result_text = f"{status_text} — Predicted CO₂ Output: {prediction:.2f}"

        return jsonify({
            "status": "success",
            "prediction": round(prediction, 2),
            "category": category,
            "status_text": status_text,
            "result_text": result_text,
            "color_class": color_class,
            "color_hex": color_hex,
            "cii_rating": cii_rating,
            "eco_impact": {
                "annual_est_tonnes": round(annual_est_emissions_tonnes, 2),
                "trees_offset": trees_offset,
                "cars_offset": cars_offset,
                "homes_offset": homes_offset
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Calculation failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

