# GreenWaves — Ship CO₂ Emission Intelligence Dashboard

GreenWaves is a modern, high-fidelity web dashboard that leverages predictive machine learning models to analyze vessel efficiency metrics, estimate Carbon Intensity Indicator (CII) ratings, and simulate maritime carbon offsets.

---

## 🚀 Key Features

* **Vessel Carbon Calculator**: Predicts the vessel's CO₂ emission index (g CO₂ per nautical mile) using a Scikit-Learn regression model based on active vessel metrics.
* **IMO CII Rating Estimation**: Instantly calculates and displays the vessel's operational Carbon Intensity Indicator rating (A–E bands) according to the latest IMO 2023 efficiency standards.
* **Vessel Type Presets**: Provides one-click Quick-Fill presets for Bulk Carrier, General Cargo, Reefer, and Ro-Ro ships to streamline evaluation.
* **Interactive Sandbox Simulator**: A client-side simulator using maritime speed-power physics equations to project fuel and emissions instantly based on speed, distance, and tuning sliders.
* **Logs & Analytics Hub**: Uses `localStorage` to save historical calculations, rendering line charts of carbon trends and benchmark comparisons against global typical averages using Chart.js.
* **Print PDF Reports**: Allows operators to download clean, formatted emission reports directly from the browser.
* **Modern Glassmorphic UI**: Features an immersive ocean-dark theme with responsive card grids and smooth CSS transitions.

---

## 🛠️ Tech Stack

* **Backend**: Python 3.11, Flask, Gunicorn, NumPy, Joblib, Pickle
* **Machine Learning**: Scikit-Learn (Linear Regression model & LabelEncoder preprocessing)
* **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism styling), Bootstrap 5, Bootstrap Icons
* **Data Visualization**: Chart.js (Gauge charts, trend lines, and bar charts)
* **DevOps**: Docker, Render Blueprint, Git / GitHub

---

## 🔧 Machine Learning Input Alignment

* **Bug Fix**: Resolved a critical production shape mismatch crash where a 9-feature array was being passed to a model fitted with 5 features.
* **Corrected Pipeline**: The app is updated to extract and pass exactly the 5 features in their correct trained order:
  1. `Technical efficiency`
  2. `Total CO₂ emissions [m tonnes]`
  3. `Annual average Fuel consumption per distance [kg / n mile]`
  4. `Time spent at sea [hours]`
  5. `Ship type_encoded`

---

## 💻 Local Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/preranadas03/CO2-Emission-Prediction.git
   cd "Ship Carbon footprint"
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🐳 Containerization & Hosting

* **Docker**: Run the application in a lightweight container using the provided `Dockerfile`:
  ```bash
  docker build -t greenwaves-predictor .
  docker run -p 5000:5000 greenwaves-predictor
  ```
* **Render Deployment**: Supported via Infrastructure-as-Code Blueprint (`render.yaml`) or Docker configurations. See [DEPLOYMENT.md](file:///c:/Users/prera/OneDrive/Desktop/Projects/Ship%20Carbon%20footprint/DEPLOYMENT.md) for full hosting walkthroughs.
