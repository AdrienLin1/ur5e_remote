import numpy as np

# 读取.npy文件
array_data = np.load('/home/yhx/wzr/Dataset_Collection/demo/23/traj/traj_17.npy')

# 确认数组形状（应该是(8,)）
print("原始数组:", array_data)
print("数组形状:", array_data.shape)



# 修改第3个元素（索引2），减去0.01
array_data[2] -= 0.02  # 等价于 array_data[2] = array_data[2] - 0.01

# 保存修改后的数组
np.save('/home/yhx/wzr/Dataset_Collection/demo/23/traj/traj_17.npy', array_data)

    