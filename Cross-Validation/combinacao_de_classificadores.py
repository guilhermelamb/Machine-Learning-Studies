import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


#loading classifiers
svm = pickle.load(open('svm_classifier.sav', 'rb'))
random_forest = pickle.load(open('random_forest_classifier.sav', 'rb'))
mlp = pickle.load(open('mlp_classifier.sav', 'rb'))

#creating new entry
new_entry = [[50000, 40, 5000]]
new_entry = np.asanyarray(new_entry)
new_entry = new_entry.reshape(-1, 1)
scaler = StandardScaler()
new_entry = scaler.fit_transform(new_entry)
new_entry = new_entry.reshape(-1, 3)

#predicting
svm_prediction = svm.predict(new_entry)
random_forest_prediction = random_forest.predict(new_entry)
mlp_preiction = mlp.predict(new_entry)

client_pays = 0
client_doesnt_pay = 0

if svm_prediction == 1:
    client_pays += 1
else:
    client_doesnt_pay += 1
    
if random_forest_prediction == 1:
    client_pays += 1
else:
    client_doesnt_pay += 1
    
if mlp_preiction == 1:
    client_pays += 1
else:
    client_doesnt_pay += 1
    
if client_pays > client_doesnt_pay:
    print('Client will pay his debt')
else:
    print('Client will not pay his debt')
    
