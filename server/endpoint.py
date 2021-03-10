from server import app

@app.route("/")
def home_page():
    return "Hello world"