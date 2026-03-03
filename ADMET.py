# 修改点：将 ADMET 改为 ADME
from tdc.single_pred import ADME

print("正在开始下载 Caco-2 数据...")

# 实例化 ADME 类
data = ADME(name = 'Caco2_Wang')

# 获取数据划分
split = data.get_split()
train, valid, test = split['train'], split['valid'], split['test']

# 查看数据
print("数据前5行：")
print(train.head())

# 保存为 CSV
train.to_csv('caco2_train.csv', index=False)
print("下载完成，已保存为 caco2_train.csv")