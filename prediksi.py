import pickle
filename = 'model-svm-TFIDF-ChiSquare-rbf-C=10.0-gamma=scale.pickle'
file_parts = filename.split('-')
fs_label = file_parts[3]
text = ['pilkada akan dimenangkan oleh anies']

with open(filename, 'rb') as fin:
    if fs_label == "None":
        vectorizer, clf = pickle.load(fin)
        extraction_text_vectors = vectorizer.transform(text)
    elif fs_label == "ChiSquare":
        vectorizer, ch2, clf = pickle.load(fin)
        extraction_text_vectors = vectorizer.transform(text)
        extraction_text_vectors = ch2.transform(extraction_text_vectors)
    elif fs_label == "CFS":
        vectorizer, cfs, clf = pickle.load(fin)
        extraction_text_vectors = vectorizer.transform(text)
        extraction_text_vectors = cfs.transform(extraction_text_vectors)

y_pred = clf.predict(extraction_text_vectors)

print(y_pred)
if y_pred:
    sentimen_svm = 'Tweet positif'
else:
    sentimen_svm = 'Tweet negatif'
print('Teks\t\t: ', text[0])
print('Sentimen\t: ', sentimen_svm)
