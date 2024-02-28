from model.CustomLITE import LITE
from utils.helper import *

df = read_dataset("ArrowHead")

# print(len(set(df[1])))
model = LITE("outputs",len(df[0]),len(set(df[1])))

hist = model.compile()