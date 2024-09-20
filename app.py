from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging


application = Flask(__name__)
app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'POST':
        # Extract the comment from the form input
        comment = request.form.get('comment')

        if not comment:
            return render_template('index.html', results='Error: No comment provided')

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()

        try:
            # Create a CustomData object to hold the comment data
            data = CustomData(comment=comment)

            # Perform the prediction
            pred = predict_pipeline.predict(data)  # Pass CustomData object
            
            # Interpret the result (positive/negative comment)
            result = 'Positive' if pred[0] == 1 else 'Negative'
        except Exception as e:
            result = f'Error: {str(e)}'

        return render_template('index.html', results=result)
    else:
        return render_template('index.html')

@app.route('/predictAPI', methods=['POST'])
@cross_origin()
def predict_api():
    if request.method == 'POST':
        try:
            # Extract comment from the JSON request
            data = request.get_json()
            comment = data.get('comment')

            if not comment:
                return jsonify({'error': 'No comment provided'}), 400

            # Initialize CustomData with the input comment
            data = CustomData(comment=comment)

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Perform the prediction
            pred = predict_pipeline.predict(data)  # Pass CustomData object
            logging.info('predicted value is',pred)
           
            # Interpret the result (positive/negative comment)
            result = 'Positive' if pred[0] == 0 else 'Negative'

            return jsonify({'sentiment': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
