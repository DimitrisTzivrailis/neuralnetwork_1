import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import random
alphabet = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
a = np.random.random((22,22))
b = (a+a.T)/2
c = np.insert(b,22,0,axis=1)
J_weights = np.insert(c.T,22,0,axis=1)
y_target = np.random.randint(0,2,size=(1003, ))
print(J_weights[22][5])
print(J_weights[5][22])
def tyxaios(s):
    lst=[]
    for i in range(0,22):
        lst.append(i)
    lst.remove(s)
    return np.random.choice(lst)

def delta_function(x,y):
    if x == y:
        return 1
    else:
        return 0
def monte_carlo(sequence, beta):
    for i in range(0,95):
        a = np.random.randint(0, 95)
        s = sequence[a]
        s_new = tyxaios(s)
        #neighbours = J_weights[sequence[a]][sequence[(a - 1) % 95]] + J_weights[sequence[(a + 1) % 95]][sequence[a]]
        neighbours = (delta_function(s_new,sequence[(a - 1) % 95])-delta_function(s,sequence[(a - 1) % 95]))+(delta_function(s_new,sequence[(a + 1) % 95])-delta_function(s,sequence[(a + 1) % 95]))
        coupling_energy = -neighbours

        if coupling_energy < 0:
            #new_s = np.random.randint(0,21)
            #while new_s == s:
            #    new_s = np.random.randint(0,21)
            #s = new_s
            s = s_new

        elif np.random.uniform(0.,1.) < np.exp(-beta * coupling_energy):
            # new_s = np.random.randint(0, 20)
            # while new_s == s:
            #     new_s = np.random.randint(0, 20)
            # s = new_s
            s = s_new
        sequence[a] = s

    return sequence



generated_sequences = []
beta = 100
for j in range(0,1003):
    random_sequence = []
    for i in range(0,95):
        n = random.randint(0,21)
        random_sequence.append(n)
    for k in range(0,50):
        monte_carlo(random_sequence,beta)
    generated_sequences.append(monte_carlo(random_sequence,beta))

print(np.array(generated_sequences).shape)
print(np.array(y_target).shape)
print(y_target)
#monte_carlo(random_sequence,beta=2)
#print(random_sequence)
X_data = []
for new_generated_sequence in generated_sequences:
    vector_array = []
    for number in new_generated_sequence:
        vector = np.zeros((21,),dtype=int)
        if not number == 21:
            vector[number] = 1
        vector_array.append(vector)
    X_data.append(np.concatenate(vector_array,axis=0))
print(np.array(X_data).shape)

# pca = PCA(n_components=2)
# pca.fit(X_data)
# x_pca = pca.transform(X_data)
#
# plt.figure(1,figsize=(8,6))
# plt.scatter(x_pca[:,0],x_pca[:,1],c=np.array(y_target))
# plt.xlabel('first principal component')
# plt.ylabel('second principal component')
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(X_data,y_target,test_size=0.3, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)

plt.figure(1)
plt.imshow(confusion_matrix(y_test,predictions),cmap='Blues', interpolation='nearest')
print(confusion_matrix(y_test,predictions))
plt.colorbar()
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()
