from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load trained model using joblib
try:
    model = joblib.load("house_price_model_lzma.pkl")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model could not be loaded: {e}")
    model = None

# ✅ Define the exact feature order
FEATURES = ['id', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
            'sqft_lot15', 'year', 'month', 'day']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model is not loaded. Please check the model file."

    try:
        # ✅ Extract input data
        input_data = []
        for feature in FEATURES:
            value = request.form.get(feature)
            if value is None or value == "":
                return f"❌ Error: Missing input for {feature}"
            input_data.append(float(value))

        # ✅ Convert to NumPy array (reshaped for model)
        input_array = np.array(input_data).reshape(1, -1)

        # ✅ Check feature size before prediction
        if input_array.shape[1] != len(FEATURES):
            return f"❌ Error: Expected {len(FEATURES)} features, but got {input_array.shape[1]}"

        # ✅ Make prediction
        predicted_price = model.predict(input_array)[0]

        return render_template("result.html", price=round(predicted_price, 2))

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
