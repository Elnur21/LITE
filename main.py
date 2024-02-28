from model.CustomLITE import LITE
from utils.helper import *

df = read_dataset("ArrowHead")

# print(len(set(df[1])))
model = LITE("outputs",len(df[0].T),len(set(df[1])))

model.compile()

X_train = df[0].reshape(df[0].shape[0], df[0].shape[1], 1)
X_test = df[2].reshape(df[2].shape[0], df[2].shape[1], 1)

hist = model.fit(X_train,df[1],X_test,df[3])