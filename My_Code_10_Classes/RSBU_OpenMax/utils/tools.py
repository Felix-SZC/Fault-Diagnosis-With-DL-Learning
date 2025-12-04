import numpy as np
import torch

# 用于动态调整优化器的学习率
def adjust_learning_rate(optimizer, epoch, config):
    '''
    optimizer: 优化器对象
    epoch: 当前的训练轮次（epoch）
    args: 包含超参数的对象，包括学习率调整策略和初始学习率。
    '''
    # lr = config['training']['learning_rate'] * (0.2 ** (epoch // 2))
    if config['training']['lradj'] == 'type1':# 定义一种学习率调整策略：学习率在每个 epoch 后乘以 0.5
        lr_adjust = {epoch: config['training']['learning_rate'] * (0.8 ** ((epoch - 1) // 1))}
    elif config['training']['lradj'] == 'type2': # 定义了一组特定的 epoch 和对应的学习率。这种策略允许在特定的 epoch 切换到一个预定义的学习率
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]  # 如果当前 epoch 需要调整学习率，则从 lr_adjust 字典中获取新的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr # 将新的学习率 lr 设置为当前参数组的学习率,这会更新优化器，使其在下一次梯度更新时使用新的学习率。
        print('Updating learning rate to {}'.format(lr)) # 输出一条信息到控制台，告知用户学习率已经被更新为新的值