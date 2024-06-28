import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from topicminer.utils.email_text_processing import read_json_to_dataframe


from ...config import (
    PROCESSED_TEXT_COL, 
    CATEGORIES_COL,
    TOKEN_FILTER_NO_BELOW, 
    TOKEN_FILTER_NO_ABOVE
    )

emails_df = read_json_to_dataframe(columns=[PROCESSED_TEXT_COL, CATEGORIES_COL])

def preprocess_data(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(df['processed_text']).toarray()
    labels = df['email_categories']
    return features, labels

def split_data(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=test_size, random_state=42)

def rebalance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    return X_train_bal, y_train_bal

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return cm, fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Load and process your JSON data here
data = load_data(json_data)  # Replace json_data with your actual JSON data
features, labels = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(features, labels)
X_train_bal, y_train_bal = rebalance_data(X_train, y_train)
classifier = train_classifier(X_train_bal, y_train_bal)
cm, fpr, tpr, roc_auc = evaluate_model(classifier, X_test, y_test)
plot_roc_curve(fpr, tpr, roc_auc)