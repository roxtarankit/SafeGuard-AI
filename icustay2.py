import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


warnings.filterwarnings("ignore")

class IcuStayPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.model = None

    def loadDataset(self):
        self.df = pd.read_csv('ICUSTAY_sorted_5000.csv')

    def datasetInformation(self):
        print('--- Understanding of Dataset ---')
        print(self.df.shape)
        print(self.df.columns)
        print(self.df.dtypes)
        print(self.df.head()) 

    def statsAnalysis(self):
        # Display basic statistics
        print('--- Basic Statistics ---')
        print(self.df.describe())
        print(self.df.info())
        print('--- Missing Values ---') 
        print(self.df.isnull().sum())

    def processDataset(self):
        print(f"Initial shape: {self.df.shape}")
        self.df.dropna(subset=['FIRST_CAREUNIT', 'LAST_CAREUNIT',
                          'FIRST_WARDID', 'LAST_WARDID', 'AGE', 'GENDER', 'LOS'], \
                            inplace=True)
        
        # Binary target: LOS > 10 days
        self.num_days = 10
        self.df['stay_over_days'] = (self.df['LOS'] > self.num_days).astype(int)

        # Encode categorical
        for col in ['FIRST_CAREUNIT', 'LAST_CAREUNIT']:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        # Encode gender (binary)
        self.df['GENDER'] = LabelEncoder().fit_transform(self.df['GENDER'])

        self.features = ['FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID', 'AGE', 'GENDER']
        self.X = self.df[self.features]
        self.y = self.df['stay_over_days']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

    def modelTraining(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Accuracy
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\nAccuracy: {acc:.3f}")
        print(classification_report(self.y_test, y_pred, digits=3))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=10 days", ">10 days"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Random Forest Confusion Matrix")
        plt.grid(False)
        plt.show()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc = roc_auc_score(self.y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Feature Importance
        feat_importance = pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)
        sns.barplot(x=feat_importance.values, y=feat_importance.index, color='orange')
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()

    def allModelTraining(self):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Support Vector Machine": SVC(probability=True, random_state=42)
        }

        for name, model in models.items():
            print(f"\n=== {name} ===")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            # Accuracy
            acc = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy: {acc:.3f}")
            print(classification_report(self.y_test, y_pred, digits=3))

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=10 days", ">10 days"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{name} - Confusion Matrix")
            plt.grid(False)
            plt.show()

            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f"{name} - ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            plt.show()

    def predictLos(self):
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train the model
        pipeline.fit(self.X_train, self.y_train)
        joblib.dump(pipeline,'RF_Reg.pkl')
        # Predictions and evaluation
        y_pred = pipeline.predict(self.X_test)

        # print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        # print("R² Score:", r2_score(self.y_test, y_pred))

        # Example prediction
        sample = self.X_test.iloc[0]
        predicted_los = pipeline.predict([sample])[0]
        #
        # print(f"Predicted ICU Stay Length (days): {predicted_los:.2f}")

    def predictIcuStay(self):
        model = joblib.load('RF_Reg.pkl')
        input_data = {
            'FIRST_CAREUNIT': ['MICU'],
            'LAST_CAREUNIT': ['MICU'],
            'FIRST_WARDID': [101],
            'LAST_WARDID': [201],
            'AGE': [30],
            'GENDER': ['F']
        }

        df = pd.DataFrame(input_data)

        # Step 3: Encode categorical variables
        # Encoding GENDER: M → 1, F → 0
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

        # Encoding FIRST_CAREUNIT and LAST_CAREUNIT using manual mapping or one-hot encoding
        # Use the same encoder/mapping used during training
        careunit_mapping = {
            'MICU': 0,
            'SICU': 1,
            'CCU': 2,
            'TSICU': 3,
            'CSRU': 4
        }
        df['FIRST_CAREUNIT'] = df['FIRST_CAREUNIT'].map(careunit_mapping)
        df['LAST_CAREUNIT'] = df['LAST_CAREUNIT'].map(careunit_mapping)

        # Step 4: Predict
        prediction = model.predict(df) * self.num_days
        days = round(prediction[0] * 3, 2)
        hrs = round(days * 24, 3) 
        
        # print(f"Predicted ICU Stay: {prediction[0]:.2f} days ({prediction[0]*24:.1f} hours)")
        print(f"Predicted ICU Stay: {days} days ({hrs} hours)")
        print
    def runPipeLine(self):
        self.loadDataset()
        self.datasetInformation()
        self.statsAnalysis()
        self.processDataset()
        #self.modelTraining()
        self.allModelTraining()
        self.predictLos()
        self.predictIcuStay()
        
obj = IcuStayPredictor()
obj.runPipeLine()