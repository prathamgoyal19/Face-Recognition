from numpy import load
import math
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
data=load('embeddings_archive.npz')
X=data['arr_0']
y=data['arr_1']

train_X, test_X, train_y, test_y=train_test_split(X, y, test_size=0.33, random_state=1)
in_encoder = Normalizer(norm='l2')
train_X=in_encoder.transform(train_X)
test_X=in_encoder.transform(test_X)

out_encoder = LabelEncoder()
out_encoder.fit(train_y)
train_y = out_encoder.transform(train_y)
test_y = out_encoder.transform(test_y)

model = SVC(kernel='linear', probability=True)
model.fit(train_X, train_y)
file_name='svm_model.sav'
pickle.dump(model, open(file_name, 'wb'))
pickle.dump(in_encoder,open("in_encoder.pkl","wb"))
pickle.dump(out_encoder,open("out_encoder.pkl","wb"))