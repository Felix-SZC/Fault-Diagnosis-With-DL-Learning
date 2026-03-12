function coefMatrix = wpcoefMatrix(signal)
%WPCOEFMATRIX 对输入信号做 6 层 db1 小波包分解，返回系数矩阵
%   输入：
%       signal     — 1×N 或 N×1 向量
%   输出：
%       coefMatrix — 2^6×(N/2^6) 的矩阵，第 k 行为第 k−1 号子带的系数

    %—— 参数定义 ————————————————————————————————————————————————
    wavelet = 'db1';      % 小波基
    level   = 6;          % 分解层数

    %—— 校验信号长度是否可分解 ———————————————————————————————
    N = numel(signal);
    if mod(N, 2^level)~=0
        error('信号长度 (%d) 必须能被 2^%d (%d) 整除。', N, level, 2^level);
    end

    %—— 小波包分解 ————————————————————————————————————————————
    wpTree = wpdec(signal, level, wavelet);

    %—— 提取子带系数 —————————————————————————————————————————
    numPackets   = 2^level;      % 子带总数
    lenPerPacket = N/numPackets; % 每个子带系数长度
    coefMatrix   = zeros(numPackets, lenPerPacket);

    for k = 1:numPackets
        coefMatrix(k, :) = wpcoef(wpTree, [level, k-1]);
    end
end