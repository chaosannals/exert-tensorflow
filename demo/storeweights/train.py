from importlib import import_module

m = import_module('model')

model = m.WeightModel()
model.summary()  # 显示模型

# 使用新的回调训练模型
model.fit()# 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。
