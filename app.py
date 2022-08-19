import os
from flask import Flask, request
from flask_cors import CORS
from transformers import pipeline
from api_keys import sentiment_api_key


app = Flask(__name__)

CORS(app)

app.logger.info("Starting app.")

# Download the sentiment model.
sentiment_task = pipeline(
    'sentiment-analysis', model='./sentiment_model', tokenizer='./sentiment_model')

app.logger.info("Loaded model.")


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/query", methods=['POST'])
def embed():
    body = request.get_json()

    if 'text' not in body or 'api_key' not in body:
        return {"error": "missing text"}, 400

    api_key = body['api_key']

    if api_key != sentiment_api_key:
        return {"error": "invalid api key"}, 400

    result = sentiment_task(body['text'])

    return {"result": result}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
