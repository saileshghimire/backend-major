from flask import Flask, request, jsonify, Response
import ultralytics
from ultralytics import YOLO
# from flask_cors import CORS
from PIL import Image
import io
 
 

app = Flask(__name__)
# CORS(app)
 # Initialize your YOLOv8 model here
try:
    yolo_model = YOLO("best.pt")
    print("YOLOv8 model initialized successfully")
except Exception as e:
    print(f"Error initializing YOLOv8 model: {e}")
    yolo_model = None


@app.route('/data', methods=['GET'])
def get_data():
    # Simulating an asynchronous operation
    data = {'data': 'Hello from Flask API!'}
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from the request
        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        
        # Perform YOLOv8 inference
        results = yolo_model.predict(image)
        
        # Process results as needed
        # ...
        result = results[0]
        box = result.boxes[0]

        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(), 2)

        return jsonify({"success": True, 'results': class_id, 'Probability':conf})
    except Exception as e:
        return jsonify({"success": False, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
