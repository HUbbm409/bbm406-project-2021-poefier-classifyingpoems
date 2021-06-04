## Hacettepe University Department of Computer Engineering BBM406 - Fundamentals of Machine Learning Term Project, Spring 2021

### Team Members:
Atakan Yüksel 21627892<br>
Ceren Korkmaz 21995445<br>
Alihan Karatatar 21904324<br>

### Subject: Poem Classification with Machine Learning and BERT
We are building learning models to predict a poem's age (modern or renaissance) and poem's type (nature, love or mythology & folklore).

### shallow_learning.py
In shallow_learning.py, we are using<br>
RandomForestClassifier(n_estimators=100),<br>
LogisticRegression(solver='liblinear', random_state=15),<br>
MultinomialNB(),<br>
KNeigborsClassifier(n_neighbors=3, metric='euclidean'),<br>
DecisionTreeClassifier(),<br>
SVC(kernel='rbf)

It has the following requirements:
```python
pip install numpy
pip install -U scikit-learn
pip install pandas
```
or using conda:
```python
conda install -c anaconda numpy
conda install -c conda-forge scikit-learn
conda install -c anaconda pandas
```

shallow_learning.py can be run as shown below:
```python
python shallow_learning.py <file_path> <age_or_type> <number_of_tests>
```

<file_path> is the path of the .csv file<br>
<age_or_type> is either age or type<br>
<number_of_tests> is an integer value that decides how many times the algorithms will run to produce accuracy.<br>

Example execution:
```python
python shallow_learning.py all.csv age 20
```

The accuracy results will be outputed to console window.

### deep_learning_bert.py
We also tried to classify the said poems using deep learning, BERT to be exact. Bidirectional Encoder Representations from Transformers (BERT) is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT is a pre-trained model that understands language, we are adding a final outer layer so it can use its understanding of language to predict poem ages and types.

It has the following requirements:
```python
pip install numpy
pip install pandas
pip install -U scikit-learn
pip install transformers
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
or using conda:
```python
conda install -c anaconda numpy
conda install -c anaconda pandas
conda install -c conda-forge scikit-learn
conda install -c conda-forge transformers
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

deep_learning_bert.py can be run as shown below:
```python
python deep_learning_bert.py <file_path> <age_or_type> <learning_rate> <number_of_epochs>
```

<file_path> is the path of the .csv file<br>
<age_or_type> is either age or type<br>
<learning_rate> is a float value that supports scientific notation. Suggested value is 2e-5
<number_of_epochs> is an integer. Suggested value is 25.

Example execution:
```python
python deep_learning_bert.py all.csv age 2e-5 25
```

Epoch round, accuracy and training loss will be outputed to console.

### Conclusion
Using shallow_learning.py and deep_learning_bert.py we achieved the following accuracy results:

Method | Type | Age
--- | --- | ---
K-Nearest Neighbors | 60% | 93%
Weighted-K-Nearest Neighbors | 58% | 90%
Logistic Regression | 74% | 93%
Support Vector Machine | 71% | 96%
Naive Bayes | 65% | 75%
Decision Tree | 60% | 82%
Random Forest | 64% | 92%
BERT | 76% | 98%

BERT is the best predictor in both cases by 2%.
