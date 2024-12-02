from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ph = float(request.form['ph'])
        temperature = int(request.form['temperature'])
        taste = int(request.form['taste'])
        odor = int(request.form['odor'])
        fat = int(request.form['fat'])
        turbidity = int(request.form['turbidity'])
        color = int(request.form['color'])

        # # print to console
        # print(ph, temperature, taste, odor, fat, turbidity, color)

        prediction = model.predict([[ph, temperature, taste, odor, fat, turbidity, color]])[0]
        return render_template('index.html', ph=ph, temperature=temperature, taste=taste, odor=odor, fat=fat, turbidity=turbidity, color=color, prediction=prediction)


if __name__ == '__main__':
    app.run(port=8080, debug=True) 