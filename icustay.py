import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

class IcuStay:
    def readDataset(self):  
        # Load dataset
        self.df = pd.read_csv('ICUSTAY_sorted_5000.csv')
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

    def distributionAnalysis(self):
        # Distribution of length of stay (LOS)
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")
        plt.grid(True)
        sns.histplot(self.df['LOS'], bins=30, kde=True)
        plt.title('Distribution of Length of Stay (LOS)')
        plt.xlabel('Length of Stay (days)')
        plt.ylabel('Frequency')
        plt.savefig('graphs/LOS_distribution.png')
        plt.show()
            
    def frequencyAnalysis(self):
        # Frequency of categorical variables
        i = 1
        categorical_cols = ['FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID']
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")
        plt.grid(True)  
        plt.suptitle('Frequency of Categorical Variables')
        plt.subplot(2,2,1)
        sns.countplot(data=self.df, x='FIRST_CAREUNIT', order=self.df['FIRST_CAREUNIT'].value_counts().index)
        plt.title('Frequency of FIRST_CAREUNIT')    
        plt.xticks(rotation=90)
        plt.subplot(2,2,2)
        sns.countplot(data=self.df, x='LAST_CAREUNIT', order=self.df['LAST_CAREUNIT'].value_counts().index)
        plt.title('Frequency of LAST_CAREUNIT')
        plt.xticks(rotation=90)
        plt.subplot(2,2,3)
        sns.countplot(data=self.df, x='FIRST_WARDID', order=self.df['FIRST_WARDID'].value_counts().index)
        plt.title('Frequency of FIRST_WARDID')
        plt.xticks(rotation=90)
        plt.subplot(2,2,4)
        sns.countplot(data=self.df, x='LAST_WARDID', order=self.df['LAST_WARDID'].value_counts().index)
        plt.title('Frequency of LAST_WARDID')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('graphs/Frequency_analysis.png')
        plt.show()

    def dropMissingData(self):
        # Drop rows with missing values in specific columns
        print('--- Dropping Missing Values ---')
        self.df = self.df.dropna(subset=['FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID'])
        # Drop rows with missing LOS
        self.df = self.df.dropna(subset=['LOS'])
        print('--- Shape after dropping missing values ---')
        print(self.df.shape)
        print(self.df.isnull().sum())       

    def createBinaryTarget(self):
        # Create binary target: 1 if LOS > 10 days, else 0
        self.df['stay_over_days'] = (self.df['LOS'] > 10).astype(int)
        print('--- Binary Target Created ---')
        print(self.df['stay_over_days'].value_counts())
        print(self.df.columns)
        print(self.df.head(10))

    def featureSelection(self):
        # Select features
        features = ['FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID']
        self.X = self.df[features]
        self.y = self.df['stay_over_days']
        print('--- Features Selected ---')
        print(self.X.head()) 
        print(self.y.head())

    def encodeCategoricalColumns(self):   
        # Encode categorical columns
        for col in ['FIRST_CAREUNIT', 'LAST_CAREUNIT']:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
        print('--- Categorical Columns Encoded ---')
        print(self.X.head())    
        
    def splitDataset(self):    
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def modelTraining(self):
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print('--- Model Training Completed ---')
        # Predict and evaluate
        self.y_pred = self.model.predict(self.X_test)
        print('--- Accuracy ---')   
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(accuracy)
        print(classification_report(self.y_test, self.y_pred, digits=3))
        cm = confusion_matrix(self.y_test, self.y_pred)
        print('--- Confusion Matrix ---')   
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix of Random Forest Classifier')   
        plt.savefig('graphs/RF_confusion_matrix.png')
        plt.show()
        print('--- ROC Curve ---')
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic on Random Forest Classifier')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('graphs/RF_ROC_curve.png')
        plt.show()
        print('--- Feature Importance ---')
        feature_importances = self.model.feature_importances_
        feature_names = self.X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}) 
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df,color='orange')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')   
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('graphs/feature_importance.png')
        plt.show()
        print(feature_importance_df)
        print('--- Feature Importance ---')
        rmse = np.sqrt(np.mean((self.y_test - self.y_pred) ** 2))
        print('RMSE:', rmse)    
        print('--- Model Evaluation ---')
        print('Accuracy:', accuracy_score(self.y_test, self.y_pred))    
        print('ROC AUC:', roc_auc_score(self.y_test, self.y_pred))
        print('Classification Report:\n', classification_report(self.y_test, self.y_pred))
        print('Confusion Matrix:\n', confusion_matrix(self.y_test, self.y_pred))
        print('--- Model Training Completed ---')

    def allModelTraining(self):
        # Define models
        models = []
        models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=42)))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(kernel='rbf', probability=True, random_state=42)))
        # Evaluate each model in turn   
        self.results = []
        self.names = []
        self.cms = []
        le_first = LabelEncoder()
        le_last = LabelEncoder()

        self.df['FIRST_CAREUNIT'] = le_first.fit_transform(self.df['FIRST_CAREUNIT'])
        self.df['LAST_CAREUNIT'] = le_last.fit_transform(self.df['LAST_CAREUNIT'])

        # Save both
        joblib.dump(le_first, 'le_first_careunit.pkl')
        joblib.dump(le_last, 'le_last_careunit.pkl')
        # Save after fitting!

        print('--- All Models Training ---')
        for name, model in models:
            print('--- Model Training ---')
            print(name)
            # Fit the model
            model.fit(self.X_train, self.y_train)
            # Predict and evaluate
            self.y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, self.y_pred)
            self.names.append(name)
            self.results.append(accuracy)
            joblib.dump(model, name+'.pkl')
            print(classification_report(self.y_test, self.y_pred, digits=3))
            cm = confusion_matrix(self.y_test, self.y_pred)
         
            self.cms.append(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix of ' + name + ' Random Forest Classifier')   
            plt.savefig('graphs/'+name+'_confusion_matrix.png')
            plt.show()
              
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic on '+name+'  Classifier')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.savefig('graphs/'+name+'_ROC_curve.png')
            plt.show()
            rmse = np.sqrt(np.mean((self.y_test - self.y_pred) ** 2))
                
            print('--- Model Evaluation '+name+' ---')
            print('Accuracy:', accuracy_score(self.y_test, self.y_pred))    
            print('ROC AUC:', roc_auc_score(self.y_test, self.y_pred))
            print('Classification Report:\n', classification_report(self.y_test, self.y_pred))
            print('Confusion Matrix:\n', confusion_matrix(self.y_test, self.y_pred))
            print('RMSE:', rmse)
            print('--- Model Training Completed ---')

    def featureImportance(self):        
        print('--- Feature Importance ---')
        self.model = DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)
        feature_importances = self.model.feature_importances_
        feature_names = self.X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}) 
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df,color='orange')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')   
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('graphs/feature_importance.png')
        plt.show()
        print(feature_importance_df)
        print('--- Feature Importance ---')
           
    def comparativeAnalysis(self):
        # Compare models
        colors = ['#A71930', '#DF4601', '#AB0003', '#003278', '#FF5910']
        plt.figure(figsize=(10, 6))
        plt.bar(self.names, self.results,color=colors)
        plt.title('Accuracy of Different Models')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig('graphs/model_comparison.png')
        plt.show()
        print('--- Model Comparison ---')
# End of class
# Function List        
obj = IcuStay()
obj.readDataset()
obj.statsAnalysis()
obj.dropMissingData()   
obj.createBinaryTarget()
obj.distributionAnalysis()
obj.frequencyAnalysis()
obj.featureSelection()
obj.encodeCategoricalColumns()
obj.splitDataset()
obj.allModelTraining()
obj.comparativeAnalysis()
obj.featureImportance()

        
        
