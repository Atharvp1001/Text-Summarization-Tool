from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import logging
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize summarization pipeline once with error handling
try:
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    logging.info("Model loaded successfully on %s", "GPU" if device == 0 else "CPU")
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    summarizer = None

def summarize_text(text):
    if not text or len(text.strip()) < 30:
        raise ValueError("Text must be at least 30 characters long")
    
    if summarizer is None:
        raise RuntimeError("Summarization model is not properly initialized")

    try:
        max_chunk = 500
        text = text.strip().replace("\n", " ")
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = []

        for chunk in chunks:
            if len(chunk.strip()) >= 30:  # BART requires at least 30 characters
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])

        if not summaries:
            raise ValueError("No valid chunks to summarize")
            
        return ' '.join(summaries)
    except Exception as e:
        logging.error("Summarization error: %s", str(e))
        raise

@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    summary = ""
    if request.method == 'POST':
        try:
            original_text = request.form['text']
            summary = summarize_text(original_text)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = "An error occurred during summarization. Please try again."
            logging.error("Summarization failed: %s", str(e))
    
    return render_template('index.html', summary=summary, error=error)

@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=str(e)), e.code
    return jsonify(error="An unexpected error occurred"), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
