import pandas as pd
import re


def clean_poem(poem):
    rx = re.compile('\W+')
    res = rx.sub(' ', poem).strip()
    return res.lower()


import sys

if len(sys.argv) != 4:
    print("Make sure you entered the arguments right.\nFirst argument: path(*.csv)\nSecond argument: prediction(age or type)\nThird argument: number of tests to get accuracy (integer)")
    print("You entered: ")
    print(sys.argv[1:])
    exit()
if not(sys.argv[2] == "age" or sys.argv[2] == "type"):
    print("Prediction can either be \'age\' or \'type\'. You entered: {}".format(sys.argv[2]))
    exit()
if not sys.argv[3].isdigit():
    print("Only integers are accepted as number of tests. You entered {}".format(sys.argv[3]))
    exit()

file_path = sys.argv[1]
predict_what = sys.argv[2]
number_of_tests = int(sys.argv[3])

poems_df = pd.read_csv(file_path)
poems_df['content'] = poems_df['content'].apply(lambda x: clean_poem(x))

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()

X = v.fit_transform(poems_df['content'])
y = poems_df[predict_what]


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

accuracies = list()
algorithm_names = ["Random Forest","Logistic Regression", "Naive Bayes", "KNN", "Decision Tree", "SVM"]
number_of_algorithms = len(algorithm_names)
for i in range(number_of_tests):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #

    clf = LogisticRegression(solver="liblinear", random_state=15)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #

    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    #

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

final_result = list()
for i in range(number_of_algorithms):
    total = 0
    for j in range(i, len(accuracies), number_of_algorithms):
        total += accuracies[j]
    print(algorithm_names[i] + " accuracy: {}".format(str((total / number_of_tests) * 100)[:5]) + "%")
    final_result.append((total / number_of_tests) * 100)