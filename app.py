from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
import nltk
from transformers import pipeline
import os

# Load Whisper model (This will take time initially)
model = whisper.load_model("base")  # You can change to "small", "medium", "large"

# Load Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Function to transcribe audio
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# API Route: Audio Transcription + Sentiment Analysis
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Save the uploaded file
    file.save(file_path)

    # Transcribe the audio file
    try:
        transcribed_text = transcribe_audio(file_path)
        sentiment_result = sentiment_pipeline(transcribed_text)
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "sentiment_analysis": sentiment_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway provides a dynamic port
    app.run(host='0.0.0.0', port=port, debug=False)


