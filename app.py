import codecs
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def write_comments(path='./data_train/train/comments.csv', data=[]):
    fw = codecs.open('./data_train/train/comments.csv', 'w', 'utf-8')
    writer = csv.writer(fw)
    writer.writerows(data)


def read_comments(path='./data_train/train/comments.csv'):
    rw = open('./data_train/train/comments.csv')
    reader = csv.reader(rw)
    result = []
    for row in reader:
        result.append(row)
    return result


def write_data_to_csv():
    train_dir_neg = './data_train/train/neg/'
    files = os.listdir(train_dir_neg)
    for file in files:
        # 0 is negative
        temp = (str(open('./data_train/train/neg/' + str(file)).read()), 0)
        comments.append(temp)

    train_dir_pos = './data_train/train/neg/'
    files = os.listdir(train_dir_pos)
    for file in files:
        # 1 is positive
        temp = (str(open('./data_train/train/pos/' + str(file)).read()), 1)
        comments.append(temp)

    write_comments(data=comments)


def remove_special_char(comments):
    processed = []
    for line in comments:
        comment = line[0]
        sentiment = line[1]
        comment = comment.replace('\n', '')
        comment = comment.replace('_', ' ')
        comment = comment.replace(',', '')
        comment = comment.replace('.', '')
        comment = comment.replace('!', '')
        comment = comment.replace('(', '')
        comment = comment.replace(')', '')
        comment = comment.replace('/', '')
        comment = comment.replace(':', '')
        comment = comment.replace('`', '')
        comment = comment.replace(' ́', '')
        comment = comment.replace(' ̃', '')
        comment = comment.replace('-', '')
        processed.append((comment, sentiment))
    return processed


def load_trained_data():
    try:
        loaded_model = joblib.load('model.pkl')
        loaded_vectorizer = joblib.load('vectorizer.pkl')

        return  loaded_model, loaded_vectorizer
    except:
        return None, None
# 0 is neg, 1 is pos

loaded_model, loaded_vectorizer = load_trained_data()

if loaded_model is None or loaded_vectorizer is None:
    comments = []
    processed = []

    comments = read_comments()
    processed = remove_special_char(comments)

    df = pd.DataFrame(comments, columns=['review', 'sentiment'])
    print(df['sentiment'].value_counts())
    # draw pie chart
    labels = 'negative', 'positive'
    sizes = [df['sentiment'].value_counts()['0'],
             df['sentiment'].value_counts()['1']]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

    #training
    x_train = df.loc[:, 'review'].values
    y_train = df.loc[:, 'sentiment'].values
    # x_test = df.loc[(len(df) / 3 * 2):, 'review'].values
    # y_test = df.loc[(len(df) / 3 * 2):, 'sentiment'].values


    vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8', decode_error='ignore')

    vectorizer.fit(x_train)

    x_train = vectorizer.transform(x_train)
    # x_test = vectorizer.transform(x_test)


    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)

    print('Score on training data is: ' + str(model.score(x_train, y_train)))

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    loaded_model, loaded_vectorizer = load_trained_data()


# print('Score on testing data is: ' + str(model.score(x_test, y_test)))

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = loaded_vectorizer.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[int(y)], proba

while (True):
    label, proba = classify(str(input())
    )
    print(label, ',probability; ', proba)