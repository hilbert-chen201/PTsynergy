import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import matplotlib

plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus'] = False
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualModel(nn.Module):
    def __init__(self, input_dim):
        super(ResidualModel, self).__init__()

        # 定义模型的结构
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # 模型的前向传播
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x + residual  # 添加残差连接
        return x
features = pd.read_csv('E:\Thesis\cell\CELL\group_OVCAR3.csv', index_col=0)

features = features.iloc[:, 1:]

features = features.values
synergy_score = pd.read_csv("E:\Thesis\cell\CELL\labelgroup_OVCAR3.csv")
labels = synergy_score.loc[:, 'synergy']
labels = labels.values.reshape(-1, 1)
train_features, test_features,train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
hidden_dim = 512  # 隐藏层维度


n_layers = 1
input_dim = features.shape[1]
model = ResidualModel(input_dim)
model = model.to(device)
# 加载保存的模型权重
saved_weights = torch.load('best_model1.pth')
model.load_state_dict(saved_weights)

# 将模型设置为评估模式
model.eval()

test_features_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(device)

# 在测试集上进行评估
test_outputs = model(test_features_tensor)
criterion = nn.MSELoss()
test_loss = criterion(test_outputs, test_labels_tensor)

print(f'Test Loss: {test_loss.item()}')

with torch.no_grad():
    test_outputs = model(test_features_tensor)

# 将预测值从张量中提取出来
predicted_values = test_outputs.cpu().numpy()

# 打印前几个预测值
print("Predicted Values:")
print(predicted_values[:5])
average_predictions = np.mean(predicted_values, axis=1, keepdims=True)
average_predictions1=pd.DataFrame(average_predictions)
test_labels1=pd.DataFrame(test_labels)
# average_predictions1.to_csv('predict1.csv',index=0)
# test_labels1.to_csv('test_labels.csv',index=0)
# 打印平均值数组
print("Average Predictions:")
print(average_predictions)
max_value1 = max(average_predictions)
min_value1 = min(average_predictions)
max_value = max(test_labels)
min_value = min(test_labels)
print("Max Value:", max_value1,max_value)
print("Min Value:", min_value1,min_value)


yTest_arr,testPredictions_arr=test_labels,average_predictions
r2 = r2_score(yTest_arr,testPredictions_arr)
print(r2)
mse = np.mean((yTest_arr - testPredictions_arr)**2)

print("Mean Squared Error (MSE):", mse)
from scipy.stats import pearsonr
yTest_flat = yTest_arr.flatten()
testPredictions_flat = testPredictions_arr.flatten()

# 使用 pearsonr 计算相关系数
p, _ = pearsonr(yTest_flat, testPredictions_flat)

print("Pearson correlation coefficient:", p)
from scipy.stats import spearmanr

# 假设 y_true 和 y_pred 是真实值和预测值的列表或数组
# 请确保它们的长度相同

# 计算斯皮尔曼相关系数
spearman_corr, p_value = spearmanr(yTest_arr, testPredictions_arr)

print("Spearman's correlation coefficient:", spearman_corr)
print("P-value:", p_value)
plt.scatter(yTest_arr,testPredictions_arr,color = 'g',marker='.')
plt.plot(yTest_arr, yTest_arr, color='black', label='x=y')
plt.text(-250,80,'R2：{:>.2f}\npearsonr：{:>.2f}\nMSE：{:>.2f}\nSpearman：{:>.2f}'.format(r2,p,mse,spearman_corr),fontsize=15)
plt.title('cancer pathway',fontsize = 20)
plt.xlabel('Observed cancer synergy score',fontsize=14)
plt.ylabel('Predicted synergy score',fontsize=14)
plt.show()
'''
xy = np.vstack([yTest_arr,testPredictions_arr])
# 例如，清理数据中的 NaN
xy= np.nan_to_num(xy)


print(xy)#  将两个维度的数据叠加
if np.any(np.isfinite(xy)):
    # 执行你的高斯核密度估计代码
    z = gaussian_kde(xy)(xy)
else:
    print("数组包含无穷大或 NaN，需要处理数据。")
    '''
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# yTest_arr, testPredictions_arr, z = yTest_arr[idx], testPredictions_arr[idx], z[idx]
#
# fig, ax = plt.subplots(figsize=(8,6))
# plt.scatter(yTest_arr, testPredictions_arr,c=z, s=20,cmap='RdYlGn_r') # cmap表示标记的颜色
# plt.plot(yTest_arr, yTest_arr, color = 'black', label = 'x=y')
# plt.colorbar()
# plt.xlabel('Observed dmso_zscore_log2',fontsize=16)
# plt.ylabel('Predicted dmso_zscore_log2',fontsize=16)
# plt.title('conc=0.13',fontdict={'size': 16})
# plt.text(-6,2,'ρ=0.70\nR2=0.49', fontsize=12)
# plt.xlim(-8, 4)
# plt.ylim(-8, 4)
# plt.show()