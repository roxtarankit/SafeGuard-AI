from flask import Flask, render_template, request
import pandas as pd
import joblib
from newkiva import KivaPrediction
import warnings

warnings.filterwarnings('ignore')
global total_cost 
global emi
TF_ENABLE_ONEDNN_OPTS=0
total_cost_p = None
total_cost_p = 0.00

app = Flask(__name__)
model = joblib.load('RF_Reg.pkl')

model1 = joblib.load('svm_model.pkl')
scaler = joblib.load('svm_scaler.pkl')
feature_columns = joblib.load('svm_feature_columns.pkl')

obj = KivaPrediction()

# Mapping used during model training
careunit_mapping = {
    'MICU': 0,
    'SICU': 1,
    'CCU': 2,
    'TSICU': 3,
    'CSRU': 4
}

# feature_columns = ['funded_amount', 'loan_amount', 'term_in_months', \
#                    'lender_count', 'male_count', 'female_count']

# expected_features = ['funded_amount', 'loan_amount', 'term_in_months', 'lender_count',
#                      'male_count', 'female_count', 'currency_ALL', 'currency_AMD', \
#                      'currency_AZN', 'currency_BOB', 'currency_EUR']

@app.route('/', methods=['GET', 'POST'])
def predict():
    cp = None
    ac = None
    total_cost = None
    emi = None
    term = None
    prediction = None
    loan = None
    if request.method == 'POST':
        first_careunit = request.form['first_careunit']
        last_careunit = request.form['last_careunit']
        first_wardid = int(request.form['first_wardid'])
        last_wardid = int(request.form['last_wardid'])
        age = int(request.form['age'])
        gender = request.form['gender']
        cp = int(request.form['cp'])
        ac = int(request.form['ac'])

        # Encode inputs
        input_data = pd.DataFrame([{
            'FIRST_CAREUNIT': careunit_mapping[first_careunit],
            'LAST_CAREUNIT': careunit_mapping[last_careunit],
            'FIRST_WARDID': first_wardid,
            'LAST_WARDID': last_wardid,
            'AGE': age,
            'GENDER': 1 if gender == 'M' else 0,
           
        }])

        prediction = round(model.predict(input_data)[0]*24, 2)  # e.g., in days
        #total_cost = 134
        total_cost = (prediction * cp) + ac
       # total_cost_p = total_cost
       # print("total_cost_p inside is {}".format(total_cost_p))
        loan="Do you want to apply for loan ?"
    return render_template('index.html', prediction=prediction,loan=loan,total_cost=total_cost)
    # total_cost_p = total_cost
    # #print("total_cost is {}".format(total_cost))
    # print("total_cost_p is {}".format(total_cost_p))

@app.route('/kiva/<pred>/<total_cost>', methods=['GET', 'POST'])
def kiva(pred,total_cost):
    from sklearn.preprocessing import LabelEncoder
    repayment_intervals = ['Weekly', 'Monthly', 'Irregular', 'Bullet']
    le = LabelEncoder()
    le.fit(repayment_intervals)
    amt = pred * 200
    if request.method == 'POST':
        # Collect data from form
        term = 0
        loan_amt = 0
        term = int(request.form['term_in_months'])
        loan_amt = int(request.form['loan_amount'])
        input_data = {
            'funded_amount': float(request.form['funded_amount']),
            'loan_amount': float(request.form['loan_amount']),
            'term_in_months': int(request.form['term_in_months']),
            'lender_count': int(request.form['lender_count']),
            'male_count': int(request.form['male_count']),
            'female_count': int(request.form['female_count']),
        }
        # Manual map
        repayment_map = {
            0.0: 'Weekly',
            0.5: 'Monthly',
            1.0: 'Irregular',
            1.5: 'Bullet'
        }
        # CALL the function
        prediction = obj.predictKivaLoan(model, scaler, input_data, feature_columns)

        # Map numeric prediction to labels
        interval_map = {0: 'Irregular', 1: 'Bullet', 3: 'Monthly'}
        repayment_interval = repayment_map.get(prediction, 'Loan Request Rejected')
        # amt = predicted_label * 200
        costa = loan_amt
        tenurea = term
        emi = costa / tenurea
        return render_template('kiva.html', prediction=repayment_interval,total_cost=total_cost,emi=emi)

    return render_template('kiva.html', prediction=None,pred=pred,total_cost=total_cost)

if __name__ == '__main__':
    app.run(debug=True)