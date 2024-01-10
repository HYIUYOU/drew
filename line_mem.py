import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
# Sample data
time_intervals = np.array([1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20])



def normalize(data):
    #max_value = max(data)
    max_value = min(v1,v2,v3,v4)
    normalized_data = [x / max_value for x in data]
    return normalized_data

chatglm1_mem = [1626, 648, 1008, 1624, 682, 1624, 804, 684, 964, 764, 1064, 766, 1064, 766, 844, 726, 670, 786, 962, 966]
chatglm2_mem = [422, 418, 418, 518,426, 434, 424, 412,452, 422, 422, 424,438, 422, 416, 430,422, 416, 430, 422]
baichuan1_mem = [4000, 3034, 3474, 3120,6362, 7357, 3996, 3434,6964, 4626, 3820, 4860,4260, 4240, 3640, 3580,3966, 3994,3030, 4260]
baichuan2_mem = [5172, 5372, 4972, 6350,4652, 5372, 5772, 4592,6352, 4812, 5012, 5372,5372, 5172, 4812, 4188,5956, 5172, 4150, 5172]

v1 = min(chatglm1_mem)
v2 = min(chatglm2_mem)
v3 = min(baichuan1_mem)
v4 = min(baichuan2_mem)



norm_chatglm1 = normalize(chatglm1_mem)
norm_chatglm2 = normalize(chatglm2_mem)
norm_baichuan1 = normalize(baichuan1_mem)
norm_baichuan2 = normalize(baichuan2_mem)




plt.plot(time_intervals, norm_chatglm1, color='#3b0f70', marker='>', linestyle='-', label='ChatGLM1', markersize=8)
plt.plot(time_intervals, norm_chatglm2, color='#de4968', marker='<', linestyle=':',label='ChatGLM2', markersize=8)
plt.plot(time_intervals, norm_baichuan1, color='#8c2981', marker='o', linestyle='--', label='Baichuan1', markersize=8)
plt.plot(time_intervals, norm_baichuan2, color='#fe9f6d', marker='x', linestyle='-.', label='Baichuan2', markersize=8,markeredgewidth=4)

# Adding labels and title
plt.xlabel('Inference Query Label')
plt.ylabel('Norm. Memory Changes')
#plt.title('Different models memory overhead.')



plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4,frameon=False)

# 适当调整图形的布局以防止切割
plt.tight_layout()
# Show the plot
plt.show()
