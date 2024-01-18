import torch
from sklearn.multioutput import MultiOutputRegressor
import GetData
import Tools
from sklearn.svm import SVR

MODEL_PATH = "model/ann-kalahai"
DATA_PATH = "database/jiduo.txt"
DEVICE_ID = "cuda:0"
LOAD_FLAG = True

torch.set_printoptions(precision=8)
TIME_STEP = 16
INPUT_SIZE = TIME_STEP * 52 * 2
OUTPUT_SIZE = 52 * 2

y_data, Range = GetData.GetDataFromTxt(DATA_PATH)

N = y_data.shape[0]
N1 = int(N * 0.7)
N2 = N - N1

print("N=" + str(N))
print("N1=" + str(N1))
print("N2=" + str(N2))


trainx = []
trainy = []
testx = []
testy = []
for i in range(TIME_STEP,N1):

    trainx.append(y_data[i-TIME_STEP:i].reshape(-1))
    trainy.append(y_data[i])
for i in range(TIME_STEP,N2):
    #for j in range(OUTPUT_SIZE):
    testx.append(y_data[N1 + i - TIME_STEP:N1 + i].reshape(-1))
    testy.append(y_data[N1 + i])

# clf = SVR(kernel = 'rbf',C = 3)
# clf = SVR(kernel='linear', C=3)
clf = MultiOutputRegressor(SVR(kernel='rbf', C=10000, gamma=0.001))

print("训练开始")
clf.fit(trainx,trainy)
print("训练完成","预测开始")
predict = clf.predict(testx)
print("预测完成")
mse = Tools.MSE(testy, predict)
print(mse)