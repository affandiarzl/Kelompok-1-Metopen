from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn import svm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# load dataset
file_dataset = 'dataset_tweet_sentiment_pilkada_DKI_2017.csv'
data = pd.read_csv(file_dataset)

# cetak informasi dataset
print(data.head())
print(data.info())

# Analisis Statistik Dataset
print('\nJumlah Data Berdasarkan Pasangan Calon:')
print(data.groupby('Pasangan Calon').size())
print('\nJumlah Data Sentiment Positive:')
dt = data.query("Sentiment == 'positive'")
print(dt.groupby('Pasangan Calon').size())
print('\nJumlah Data Sentiment Negative:')
dt = data.query("Sentiment == 'negative'")
print(dt.groupby('Pasangan Calon').size())
data['Sentiment'].value_counts().plot(kind='bar')
plt.show()

# Pembersihan Data
def remove_at_hash(sent):
    return re.sub(r'@|#', r'', sent.lower())

def remove_sites(sent):
    return re.sub(r'http.*', r'', sent.lower())

def remove_punct(sent):
    return ' '.join(re.findall(r'\w+', sent.lower()))

data['text'] = data['Text Tweet'].apply(lambda x:
                                        remove_punct(remove_sites(remove_at_hash(x))))
print(data.head())

# Label Encoder
le = preprocessing.LabelEncoder()
le.fit(data['Sentiment'])
data['label'] = le.transform(data['Sentiment'])
print(data)

# Split Data
X = data['text']
y = data['label']
# split data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25)
print(X_train[0], '-', y_train[0])

# Kontrol Parameter SVM
pKernel = ['linear', 'rbf']  # kernel SVM
pC = [0.1, 1.0, 10.0]  # nilai C (hyperplane)
pGamma = ["scale", "auto", 0.1] # nilai gamma
fe = ["TFIDF", "BoW"]  # Ekstrakksi Fitur
fs = ["None", "ChiSquare", "CFS"]  # Seleksi Fitur
table = []  # Tabel untuk menyimpan hasil

counter = 1
for ife in range(len(fe)):
    for ifs in range(len(fs)):
        for ik in range(len(pKernel)):
            for ic in range(len(pC)):
                for ig in range(len(pGamma)):
                    # Pemilihan ekstraksi fitur, 0 = TF-IDF, 1 = BoW
                    if ife == 0:
                        fe_label = fe[ife]
                        extraction_vectorizer = TfidfVectorizer()
                        extraction_train_vectors = extraction_vectorizer.fit_transform(X_train)
                        extraction_test_vectors = extraction_vectorizer.transform(X_test)
                    else:
                        fe_label = fe[ife]
                        extraction_vectorizer = CountVectorizer()
                        extraction_train_vectors = extraction_vectorizer.fit_transform(X_train)
                        extraction_test_vectors = extraction_vectorizer.transform(X_test)
                        
                    # Pemilihan seleksi fitur, 0 = None, 1 = ChiSquare, 2 = CFS
                    if ifs == 0:
                        fs_label = fs[ifs]
                    elif ifs == 1:
                        fs_label = fs[ifs]
                        ch2 = SelectKBest(chi2, k=900)
                        extraction_train_vectors = ch2.fit_transform(extraction_train_vectors, y_train)
                        extraction_test_vectors = ch2.transform(extraction_test_vectors)
                    elif ifs == 2:
                        fs_label = fs[ifs]
                        cfs = SelectKBest(f_classif, k=900)
                        extraction_train_vectors = cfs.fit_transform(extraction_train_vectors, y_train)
                        extraction_test_vectors = cfs.transform(extraction_test_vectors)

                    # Training dan Testing SVM(80 : 20)
                    svm_classifier = svm.SVC(kernel=pKernel[ik], C=pC[ic], gamma=pGamma[ig])
                    svm_classifier.fit(extraction_train_vectors, y_train)  # training
                    y_pred = svm_classifier.predict(extraction_test_vectors)  # testing

                    # Confusion Matrix
                    cnf_matrix = confusion_matrix(y_test, y_pred)
                    TN, FP, FN, TP = cnf_matrix.ravel()
                    cm = TN, FP, FN, TP

                    # Statistik hasil percobaan
                    precision = round(precision_score(y_test, y_pred), 2)
                    recall = round(recall_score(y_test, y_pred), 2)
                    accuracy = round(accuracy_score(y_test, y_pred), 2)
                    f1 = round(f1_score(y_test, y_pred), 2)

                    # Tambahkan hasil ke tabel
                    table.append([counter, fe_label, fs_label, pKernel[ik], pC[ic], pGamma[ig], extraction_train_vectors.shape[0],
                                  extraction_train_vectors.shape[1], cm, precision, recall, accuracy, f1])

                    # Simpan Model
                    filename = f'model-svm-{fe_label}-{fs_label}-{pKernel[ik]}-C={pC[ic]}-gamma={pGamma[ig]}.pickle'
                    pickle.dump(svm_classifier, open(filename, 'wb'))
                    vectorizer = extraction_vectorizer
                    vectorizer.stop_words_ = None
                    clf = svm_classifier
                    with open(filename, 'wb') as fout:
                        if ifs == 0:
                            pickle.dump((vectorizer, clf), fout)
                        elif ifs == 1:
                            pickle.dump((vectorizer, ch2, clf), fout)
                        elif ifs == 2:
                            pickle.dump((vectorizer, cfs, clf), fout)

                    # Increment number
                    counter += 1

# Cetak table
headers = ["No", "Ekstraksi Fitur", "Seleksi Fitur", "Kernel", "C", "Gamma", "Jumlah Data",
           "Jumlah Fitur", "CM(TN, FP, FN, TP)","Precision", "Recall", "Accuracy", "F1-Score"]
table_str = tabulate(table, headers, numalign="right")
print(table_str)

csv_file = "result.csv"

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Tulis header
    writer.writerow(headers)

    # Tulis data dari tabel yang diformat
    for line in table_str.splitlines()[2:]:
        row = line.split()
        writer.writerow(row)
