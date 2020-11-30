from importlib import import_module

m = import_module('model')

# 创建一个基本模型实例
model = m.WeightModel()
# 评估模型
model.evaluate()
# 加载权重
model.load_weight()
# 重新评估模型
model.evaluate()