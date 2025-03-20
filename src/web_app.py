from flask import Flask, render_template, request, jsonify
import logging
from predict import DelayPredictor
import pickle
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor
try:
    predictor = DelayPredictor()
    # Load station list from metadata
    with open('./models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
        station_encoder = metadata['label_encoders']['Station']
        station_list = station_encoder.classes_.tolist()
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    station_list = []

@app.route('/')
def home():
    return render_template('index.html', stations=station_list)

@app.route('/suggest_stations')
def suggest_stations():
    query = request.args.get('query', '').upper()
    if not query:
        return jsonify([])
    
    # Filter stations that start with the query
    suggestions = [station for station in station_list if station.upper().startswith(query)]
    return jsonify(suggestions[:10])  # Limit to 10 suggestions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        date = request.form.get('date')
        time = request.form.get('time')
        station = request.form.get('station')
        line = request.form.get('line')
        bound = request.form.get('bound')
        
        # Log the input data
        logger.info(f"Received prediction request with data: date={date}, time={time}, station={station}, line={line}, bound={bound}")
        
        # Make prediction
        result = predictor.predict_delay(
            date=date,
            time=time,
            station=station,
            line=line,
            bound=bound
        )
        
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 