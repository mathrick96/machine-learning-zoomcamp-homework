from flask import Flask, request, jsonify
import pickle


def pickle_load_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


dv = pickle_load_file('dv.bin')
model = pickle_load_file('model1.bin')



app = Flask('score_customer')

@app.route('/score_customer', methods=['POST'])
def score_customer():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    result = {'probability' : float(y_pred[0])}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)