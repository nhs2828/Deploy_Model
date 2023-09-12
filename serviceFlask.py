import pickle
import xgboost as xgb
from flask import Flask
from flask import request
from flask import jsonify
import json



with open('models/DT.bin', 'rb') as f_in:
    dv, clf_dt = pickle.load(f_in)

with open('models/RF.bin', 'rb') as f_in:
    dv, clf_rf = pickle.load(f_in)

with open('models/XGB.bin', 'rb') as f_in:
    dv, clf_xgb = pickle.load(f_in)


app = Flask('predict')

@app.route('/dt', methods = ['POST'])
def predict_dt():
    customer = request.get_json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    pred = clf_dt.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return jsonify(res)

@app.route('/rf', methods = ['POST'])
def predict_rf():
    customer = request.get_json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    pred = clf_rf.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return jsonify(res)

@app.route('/xgb', methods = ['POST'])
def predict_xgb():
    customer = request.get_json()
    customer = json.loads(customer)
    x = dv.transform([customer])
    features = dv.get_feature_names_out()
    x = xgb.DMatrix(x, feature_names=features)
    pred = clf_xgb.predict(x)[0]
    res = {
        "res": int(pred)
    }

    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6969)