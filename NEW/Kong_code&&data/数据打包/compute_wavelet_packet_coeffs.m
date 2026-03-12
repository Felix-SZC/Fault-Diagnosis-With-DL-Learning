% compute_wavelet_packet_coeffs.m
% 读取 data_sample.mat 中的 data，计算每个样本的小波包系数并保存

% 清理环境
clear; clc;

% 载入数据
load('data_sample.mat', 'data');  % data 大小应为 10800×4096

% 获取样本数
[nSamples, sigLen] = size(data);
if sigLen ~= 4096
    error('数据维度不符：每个样本应为长度 4096 的信号。');
end

% 预分配系数矩阵
% 最终矩阵 coefAll 大小为 nSamples×64×64
coefAll = zeros(nSamples, 64, 64);

% 循环计算
for i = 1 : nSamples
    % 提取第 i 个样本
    xi = data(i, :);
    % 计算 64×64 的小波包系数矩阵
    Ci = wpcoefMatrix(xi);
    % 检查输出尺寸
    if ~isequal(size(Ci), [64, 64])
        error('第 %d 个样本的 coefMatrix 输出尺寸不是 64×64。', i);
    end
    % 存入结果矩阵
    coefAll(i, :, :) = Ci;
end

% 保存结果
save('coefAll.mat', 'coefAll');

fprintf('处理完成，已生成并保存 coefAll.mat，尺寸为 %d×64×64。\n', nSamples);