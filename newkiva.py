import numpy as np
import pandas as pd
#matplotlib
import matplotlib.pyplot as plt
#train_test_split
from sklearn.model_selection import train_test_split
#StandardScaler
from sklearn.preprocessing import StandardScaler
#tensorflow
import tensorflow as tf
#metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import joblib 

warnings.filterwarnings('ignore')

cols = ['id','funded_amount','loan_amount','activity','sector','use','country',\
        'region','currency','partner_id','posted_time','disbursed_time','funded_time',\
    	'term_in_months','lender_count','tags','borrower_genders','repayment_interval','date']
# ['id', 'use', 'tags', 'country', 'date', 'region', 'partner_id'] 

class KivaPrediction:
    def readDataSet(self):
        self.df1=pd.read_csv('kiva_loans_small23_Final1.csv',names=cols)
        print(self.df1)

    def getHealthData(self):
        self.df = self.df1[self.df1['activity'] == 'Health']   
        print(self.df)
        print(len(self.df))
        self.df.to_csv('kiva_health_records.csv', index=False)

    def datasetInformation(self):
        print('---- Shape of Dataset ----')
        print(self.df.shape) 
        print('Rows=',self.df.shape[0],'Columns =',self.df.shape[1])
        print('--- Dataset Columns ---')
        print(self.df.columns)
        print('--- Dataset Columns Data Types ---') 
        print(self.df.dtypes)
        print('--- Statistical Information ---')
        print(self.df.describe())

    def datasetSamples(self):
        print("--- First 5 Records ---")
        print(self.df.head(5))
        print("--- Last 5 Records ---")
        print(self.df.tail(5))

    def dataVisualization(self):
        top_country=self.df['country'].value_counts().reset_index(name='count').head(8)
        top_country=top_country.rename(columns={'index':'country'})
        print(type(top_country))
        c1 = top_country['country'].tolist()
        d1 = top_country['count'].tolist()
        print(c1)
        print(d1)
        plt.figure(figsize=(10,5))
        plt.grid(True)
        plt.bar(c1,d1)
        plt.title('Top 8 countries that recieved funds')
        plt.xlabel('Country Name')
        plt.ylabel('No of Times Funds Received')
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.savefig('graphs/kiva_countries.png')
        plt.show()

        region=self.df['region'].value_counts().reset_index(name='count').head(8)
        region=region.rename(columns={'index':'region'})
        c1 = region['region'].tolist()
        d1 = region['count'].tolist()
        print(c1)
        print(d1)
        plt.figure(figsize=(10,5))
        plt.grid(True)
        plt.bar(c1,d1,color='red')
        plt.title('Top 8 Regions that recieved funds')
        plt.xlabel('Region Name')
        plt.ylabel('No of Times Funds Received')
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.savefig('graphs/kiva_regions.png')
        plt.show()

    def dataProcessing(self):
        top_country=self.df['country'].value_counts().reset_index(name='count').head(10)
        top_country=top_country.rename(columns={'index':'country'})
        print('---- Top Countries ----')
        print(top_country)  
        region=self.df['region'].value_counts().reset_index(name='count')
        region=region.head(10)
        region=region.rename(columns={'index':'region'})  
        print('---- Top Regions ----')
        print(region)
        print('---- Repayment Intervals ----')
        repayment_interval=self.df['repayment_interval'].value_counts().reset_index(name='count')
        repayment_interval=repayment_interval.rename(columns={'index':'Type of Repayment'})
        print(repayment_interval)
        print('--- Sectors for Funding ---')
        print(self.df['sector'].value_counts())
        print('--- Funds in Currency ---')
        currency_counts=self.df['currency'].value_counts().head(5)
        print(currency_counts)
        
    def convert_datetime(self,df,column):
        df=df.copy()
        df[column]=pd.to_datetime(df[column])
        df[column+'_y']=df[column].dt.year
        df[column+'_m']=df[column].dt.month
        df[column+'_d']=df[column].dt.day
        df=df.drop(column,axis=1)
        return df       
    
    def male_count(self,x):
        count=0
        for male in str(x).split(', '):
            if male=='male':
                count+=1
        return count
    
    def onehot_encode(self,df,columns):
        for column in columns:
            dummies=pd.get_dummies(df[column],prefix=column)
            df=pd.concat([df,dummies],axis=1)
            df=df.drop(column,axis=1)
        return df

    def female_count(self,x):
        count=0
        for female in str(x).split(', '):
            if female=='female':
                count+=1
        return count
    
    def findAndReplaceMissingValues(self):
        print('--- Missing Values in Dataset ---')
        print(self.df.isnull().sum()) 
        data=[]
        for c in cols:
            if self.df[c].isnull().sum() > 0:
                data.append(c)
                print(self.df[c].dtype)
        print('--- Columns with Missing Values in Dataset ---')
        print(data)
        print('--- Replacing the Missing Values ---')
        
        for d in data:
            if self.df[d].dtype==object:
                self.df[d].fillna(self.df[d].mode(),inplace=True)
                # print(self.df[d].mode())
            else:
                self.df[d].fillna(self.df[d].mean(),inplace=True) 
                # print(self.df[d].mean())
        print('--- Missing Values in Dataset after replacement ---')
        print(self.df.isnull().sum())

    def dataCleaning(self):
        vc = self.df['repayment_interval'].value_counts()
        print('--- Value Counts ---')
        print(vc)

    # def understandingDataset(self):
    def dataPreparation(self):
        print(self.df.columns)
        # self.df=self.df.copy()
        # #dropping the id column
        self.df=self.df.drop(['activity','id','use','tags','country','date','region','partner_id'],axis=1)
        print('---Columns----')
        print(self.df.columns)
        self.df=self.convert_datetime(self.df,'posted_time')
        for column in ['disbursed_time','funded_time']:
            self.df[column]=self.df[column].fillna(self.df[column].mode()[0])
        self.df=self.convert_datetime(self.df,'disbursed_time')
        self.df=self.convert_datetime(self.df,'funded_time')
        self.df['male_count']=self.df['borrower_genders'].apply(self.male_count)
        self.df['female_count']=self.df['borrower_genders'].apply(self.female_count)
        self.df=self.df.drop('borrower_genders',axis=1)
        y=self.df['repayment_interval']
        x=self.df.drop('repayment_interval',axis=1)
        x=self.onehot_encode(x,[column for column in x.select_dtypes('object')])
        y=y.replace({'irregular':0,'bullet':1,'monthly':3})
        print('--- X ---')
        print(x)

        #train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)
        scaler=StandardScaler()
        scaler.fit(x_train)
        x_train=pd.DataFrame(scaler.transform(x_train),columns=x_train.columns)
        x_test=pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
        # x_train,x_test,y_train,y_test=self.preprocess_inputs(self.df)
        print(x_train.shape)
        print(x_train.shape)
        print(x_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        feature_columns = list(self.x_train.columns)
        print("Number of Feature Columns:", len(feature_columns))
        print("Sample Columns:", feature_columns[:5])
        print('--- Feature Columns Saved as Pickle ---')
        print(feature_columns)
        # Save to pickle
        # joblib.dump(feature_columns, 'kiva_feature_columns.pkl')

    def allModelTraining(self):
        print("\n--- Training Multiple ML Models ---\n")
        self.accs = []
        self.names = []
        # Prepare data again if needed (assuming it's already split/scaled in dataPreparation)
        top_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

        # Train and evaluate each model
        for name, model in top_models.items():
            print(f"\n--- {name} ---")
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)

            scaler = StandardScaler()
            scaler.fit(self.x_train)

            # Transform the features
            X_train_scaled = scaler.transform(self.x_train)
            X_test_scaled = scaler.transform(self.x_test)

            # joblib.dump(model,name+'_model.pkl')
            # joblib.dump(model,name+'_scaler.pkl')

            acc = accuracy_score(self.y_test, y_pred)
            self.accs.append(acc)
            self.names.append(name)
            print(f"Accuracy: {acc:.4f}")
            print("Classification Report:\n", classification_report(self.y_test, y_pred))
            
            # Optional: Confusion Matrix Heatmap
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()    

    def comparativeAnalysis(self):
        plt.figure(figsize=(10,5))
        plt.grid(True)
        sns.set_style('darkgrid')
        plt.bar(self.names,self.accs)
        plt.show()

    def train_svm_model(self):
        from sklearn.preprocessing import LabelEncoder
        print('--- Final Train SVM Model ---')
        feature_columns = ['funded_amount', 'loan_amount', 'term_in_months', 'lender_count', 'male_count', 'female_count']
        target_column='repayment_interval'
        model_path='svm_model.pkl'
        scaler_path='svm_scaler.pkl'
        features_path='svm_feature_columns.pkl'
        data=self.df
        le = LabelEncoder()
        self.df['loan_amount'] = le.fit_transform(self.df['loan_amount'])
        """
        Trains an SVM model and saves the model, scaler, and feature column list as pickle files.

        Parameters:
        - data (pd.DataFrame): The dataset containing features and target.
        - target_column (str): Name of the column to be predicted.
        - feature_columns (list): List of feature column names.
        - model_path (str): Path to save the trained model.
        - scaler_path (str): Path to save the fitted scaler.
        - features_path (str): Path to save the feature column list.
        """

        # Step 1: Prepare features and target
        X = data[feature_columns]
        y = data[target_column]
        print(X.columns)
        print(X.head(10))

        # Step 2: Split the data (optional but recommended for validation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Only use the features your model was trained on

        model = SVC()  # You can pass parameters like kernel='linear', C=1, etc.
        model.fit(X_train_scaled, y_train)

        # Step 5: Save model, scaler, and feature columns
        with open(model_path, 'wb') as f:
            joblib.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            joblib.dump(scaler, f)
        
        with open(features_path, 'wb') as f:
            joblib.dump(feature_columns, f)

        print("✅ Model training complete. Files saved:")
        print(f"- Model: {model_path}")
        print(f"- Scaler: {scaler_path}")
        print(f"- Feature columns: {features_path}")

    def predictKivaLoan(self, model, scaler, input_data, feature_columns):
        """
        Predicts repayment interval for Kiva loan based on user input.

        Parameters:
        - model: trained ML model (e.g., RandomForestClassifier)
        - scaler: fitted StandardScaler
        - input_data: dictionary with user input data
        - feature_columns: list of features used during training

        Returns:
        - prediction: integer class (0, 1, or 3) representing repayment_interval
        """
        # Create DataFrame
        # Only use the features your model was trained on
        model_input = {k: input_data[k] for k in feature_columns}
        df_input = pd.DataFrame([model_input])
        scaled_input = scaler.transform(df_input)

        # Predict
        prediction = model.predict(scaled_input)[0]
        print("--- Prediction ---")
        print(prediction)
        # Map prediction to readable form if needed
        mapping = {0: "irregular", 1: "bullet", 3: "monthly"}
        return prediction

    def runPipeLine(self):
        self.readDataSet()      
        self.getHealthData()
        self.datasetInformation()  
        self.datasetSamples()
        self.dataVisualization()
        self.dataProcessing()
        self.findAndReplaceMissingValues()
        self.dataPreparation()
        self.dataCleaning()
        self.allModelTraining()
        self.comparativeAnalysis()
        self.train_svm_model()

        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('svm_scaler.pkl')
        feature_columns = joblib.load('svm_feature_columns.pkl')
        #'funded_amount', 'loan_amount', 'term_in_months', 'lender_count', \
        # 'male_count', 'female_count'
        input_data = {
            'funded_amount': 500,
            'loan_amount': 500,
            'term_in_months': 8,
            'lender_count': 12,
            'male_count': 1,
            'female_count': 0
            # This and others must match one-hot training values
        }
        predicted_interval = self.predictKivaLoan(model, scaler, input_data, feature_columns)
        print("Predicted Repayment Interval:", predicted_interval)
                    
# obj = KivaPrediction()
# obj.runPipeLine()

