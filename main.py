import mysql.connector
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import pandas as pd
import mariadb

# Source Code for Mysql Database
# mydb = mysql.connector.connect(
#     host = "localhost",
#     user = "root",
#     password = "",
#     database = "bensin",
# )

# Source Code for Mariadb Database
mydb = mariadb.connect(
    host = "localhost",
    user = "root",
    password = "123456",
    database = "bensin",
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM tableName")

myresult = mycursor.fetchall()

Liter = [x[0] for x in myresult]
Kilometer = [x[1] for x in myresult]

liter = np.array(Liter)
kilometer = np.array(Kilometer)


X_train, X_test, y_train, y_test = ms.train_test_split(liter, kilometer, test_size=0.2, random_state=0)

# X_train = pd.DataFrame(X_train, columns=["liter"])
# y_train = pd.DataFrame(y_train, columns=["kilomter"])
# X_test = pd.DataFrame(X_test, columns=["liter"])
# y_train = pd.DataFrame(y_train, columns=["kilomter"])

# Merubah array menjadi 1 Kolom
X_train = np.array(X_train).reshape((len(X_train), 1))
y_train = np.array(y_train).reshape((len(y_train), 1))
X_test = np.array(X_test).reshape((len(X_test), 1))
y_test = np.array(y_test).reshape((len(y_test), 1))

model = lm.LinearRegression()
model.fit(X_train, y_train)

prediksi = model.predict(X_test)

prediksi =  [x[0] for x in prediksi]
print(prediksi)