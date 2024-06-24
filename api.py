from flask import Flask, jsonify, request
import pandas as pd
from Pipeline import main


app = Flask(__name__)

@app.route('/apt_predict', methods=['POST'])
def process_prediction():
    if request.is_json:
        data = request.get_json()
        dataframe = pd.DataFrame(data)
        result = main(dataframe)
        response = {
            "Prediction" : result
        }
        return jsonify(response)
    else:
        return "Json Not recieved."

   
if __name__ == "__main__":
    from waitress import serve
    # serve(app, host="0.0.0.0", port=8081, threads=20)
    app.run(debug=False)