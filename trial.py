import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from keras.models import Sequential
from keras.layers import Dense, Dropout

# -------------------------------
# DATA
# -------------------------------
X, y = None, None
X_train, X_test, y_train, y_test = None, None, None, None

metrics = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1": {}}

# -------------------------------
# WINDOW
# -------------------------------
root = tk.Tk()
root.title("Cyber Threat Detection System")
root.geometry("1200x700")
root.configure(bg="#1e1e2f")

# -------------------------------
# STYLE
# -------------------------------
style = ttk.Style()
style.theme_use('clam')

# -------------------------------
# LEFT PANEL (CONTROLS)
# -------------------------------
left_frame = tk.Frame(root, bg="#2b2b3c", width=300)
left_frame.pack(side="left", fill="y")

title = tk.Label(left_frame, text="Cyber Threat\nDetection",
                 bg="#2b2b3c", fg="white",
                 font=("Arial", 16, "bold"))
title.pack(pady=20)

# -------------------------------
# RIGHT PANEL (OUTPUT)
# -------------------------------
right_frame = tk.Frame(root, bg="#1e1e2f")
right_frame.pack(side="right", fill="both", expand=True)

text = tk.Text(right_frame, bg="#111", fg="lime",
               font=("Consolas", 11))
text.pack(fill="both", expand=True, padx=10, pady=10)

# -------------------------------
# FUNCTIONS
# -------------------------------
def log(msg):
    text.insert(tk.END, msg + "\n")
    text.see(tk.END)

def upload():
    global X, y

    file = filedialog.askopenfilename()
    data = pd.read_csv(file)

    X = data.drop('labels', axis=1)
    y = data['labels']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = pd.get_dummies(X).astype('float32')
    X = StandardScaler().fit_transform(X)

    log(f"Loaded dataset: {file}")
    log(f"Shape: {X.shape}")

def split_data():
    global X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log(f"Train: {X_train.shape}")
    log(f"Test : {X_test.shape}")

def store(name, y_pred):
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro') * 100
    rec = recall_score(y_test, y_pred, average='macro') * 100
    f1 = f1_score(y_test, y_pred, average='macro') * 100

    metrics["Accuracy"][name] = acc
    metrics["Precision"][name] = prec
    metrics["Recall"][name] = rec
    metrics["F1"][name] = f1

    log(f"\n{name} Results:")
    log(f"Accuracy : {acc:.2f}")
    log(f"Precision: {prec:.2f}")
    log(f"Recall   : {rec:.2f}")
    log(f"F1 Score : {f1:.2f}")

# -------------------------------
# ANN
# -------------------------------
def run_ann():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=10, batch_size=64,
                        validation_data=(X_test, y_test),
                        verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    store("ANN", y_pred)

    # Graph
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("ANN Accuracy")
    plt.legend(["Train", "Test"])
    plt.show()

# -------------------------------
# CLASSIFIERS
# -------------------------------
def run_svm():
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    store("SVM", model.predict(X_test))

def run_knn():
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    store("KNN", model.predict(X_test))

def run_rf():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    store("RandomForest", model.predict(X_test))

def run_nb():
    model = BernoulliNB()
    model.fit(X_train, y_train)
    store("NaiveBayes", model.predict(X_test))

def run_dt():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    store("DecisionTree", model.predict(X_test))

# -------------------------------
# GRAPH
# -------------------------------
def plot(metric):
    names = list(metrics[metric].keys())
    values = list(metrics[metric].values())

    plt.bar(names, values)
    plt.title(metric)
    plt.show()

# -------------------------------
# BUTTON HELPER
# -------------------------------
def add_button(text, command):
    btn = tk.Button(left_frame, text=text, command=command,
                    bg="#4e4e8a", fg="white",
                    font=("Arial", 10, "bold"),
                    width=25)
    btn.pack(pady=5)

# -------------------------------
# BUTTONS
# -------------------------------
add_button("Upload Dataset", upload)
add_button("Split Data", split_data)
add_button("Run ANN", run_ann)

add_button("Run SVM", run_svm)
add_button("Run KNN", run_knn)
add_button("Random Forest", run_rf)
add_button("Naive Bayes", run_nb)
add_button("Decision Tree", run_dt)

add_button("Accuracy Graph", lambda: plot("Accuracy"))
add_button("Precision Graph", lambda: plot("Precision"))
add_button("Recall Graph", lambda: plot("Recall"))
add_button("F1 Graph", lambda: plot("F1"))

# -------------------------------
# RUN
# -------------------------------
root.mainloop()