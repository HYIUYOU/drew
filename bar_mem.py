

# 导入matplotlib.pyplot和numpy库
import matplotlib.pyplot as plt
import numpy as np

# 给定的四个数据集
chatglm1_mem = [1626, 648, 1008, 1624, 682, 1624, 804, 684, 964, 764, 1064, 766, 1064, 766, 844, 726, 670, 786, 962, 966]
chatglm2_mem = [422, 418, 418, 518, 426, 434, 424, 412, 452, 422, 422, 424, 438, 422, 416, 430, 422, 416, 430, 422]
baichuan1_mem = [4000, 3034, 3474, 3120, 6362, 7357, 3996, 3434, 6964, 4626, 3820, 4860, 4260, 4240, 3640, 3580, 3966, 3994, 3030, 4260]
baichuan2_mem = [5172, 5372, 4972, 6350, 4652, 5372, 5772, 4592, 6352, 4812, 5012, 5372, 5372, 5172, 4812, 4188, 5956, 5172, 4150, 5172]

# 计算最小值、最大值和平均值
v1 = min(min(chatglm1_mem), min(chatglm2_mem), min(baichuan1_mem), min(baichuan2_mem))
v2 = min(max(chatglm1_mem), max(chatglm2_mem), max(baichuan1_mem), max(baichuan2_mem))
v3 = min([np.mean(chatglm1_mem), np.mean(chatglm2_mem), np.mean(baichuan1_mem), np.mean(baichuan2_mem)])

# 根据这些值计算每个组的标准化值
group1 = [min(chatglm1_mem)/v1, max(chatglm1_mem)/v2, np.mean(chatglm1_mem)/v3]
group2 = [min(chatglm2_mem)/v1, max(chatglm2_mem)/v2, np.mean(chatglm2_mem)/v3]
group3 = [min(baichuan1_mem)/v1, max(baichuan1_mem)/v2, np.mean(baichuan1_mem)/v3]
group4 = [min(baichuan2_mem)/v1, max(baichuan2_mem)/v2, np.mean(baichuan2_mem)/v3]

# 设定每组数据之间的间隔
bar_width = 0.2
# 设定每个分类的标签
categories = ['Minimum', 'Maximum', 'Average']

# 计算每个柱状图的位置
index = np.arange(len(categories))

# 绘制柱状图
fig, ax = plt.subplots()
bar1 = ax.bar(index, group1, bar_width, label='ChatGLM1', color='#3b0f70')
bar2 = ax.bar(index + bar_width, group2, bar_width, label='ChatGLM2', color='#de4968')
bar3 = ax.bar(index + 2 * bar_width, group3, bar_width, label='Baichuan1', color='#8c2981')
bar4 = ax.bar(index + 3 * bar_width, group4, bar_width, label='Baichuan2', color='#fe9f6d')

# 添加图例、标题和标签
ax.set_ylabel('Normalized Value')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(categories)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

# 显示图表
plt.tight_layout()
plt.show()
