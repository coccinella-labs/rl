from flask import Flask
from harpertoken.models.model import CMAESAgent

app = Flask(__name__)

# Load the model on startup
agent = CMAESAgent.from_pretrained("harpertoken/pole")


@app.route("/")
def index():
    return "RL Optimizer is running. Model loaded successfully."


@app.route("/status")
def status():
    return {"status": "Model loaded", "model": "harpertoken/pole"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
