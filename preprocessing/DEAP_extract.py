import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from pykalman import KalmanFilter  # 用于 LDS 平滑

# 设定路径
input_folder = "/mnt/DATA-2/DEAP/"
output_folder = "/mnt/DATA-2/DEAP/Preprocessed/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 参数设定
fs = 128  # 采样率
baseline_len = 3  # 3秒
n_baseline_points = baseline_len * fs  # 384个采样点

# 滑动窗口参数：窗口大小 2秒，步长 1秒（50%重叠）
window_size_sec = 2
step_size_sec = 1
win_size_points = window_size_sec * fs  # 256个采样点
step_points = step_size_sec * fs  # 128个采样点

# 更新的频段（5个频段）
bands = [
    ('theta', 4, 7),
    ('alpha', 8, 10),
    ('slow_alpha', 8, 13),
    ('beta', 14, 29),
    ('gamma', 30, 45)
]


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Butterworth带通滤波器"""
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)


def compute_de(signal):
    """计算差分熵"""
    var = np.var(signal, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def lds_smoothing(time_series):
    """
    使用简单的随机游走模型对一维时间序列进行 LDS 平滑。
    模型假设：
      x_{t+1} = x_t + w_t
      y_t = x_t + v_t
    """
    # 构造 KalmanFilter 对象
    kf = KalmanFilter(
        transition_matrices=1,
        observation_matrices=1,
        initial_state_mean=time_series[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    smoothed_state_means, _ = kf.smooth(time_series)
    return smoothed_state_means.flatten()


# 遍历所有 .mat 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".mat"):
        file_path = os.path.join(input_folder, file_name)
        print(f"Processing: {file_name}")

        # 读取数据
        mat_data = sio.loadmat(file_path)

        # 假设 EEG 数据存储在 mat_data['data']，形状 = (n_trials, n_channels, n_timepoints)
        if 'data' not in mat_data:
            print(f"Skipping {file_name}, no 'data' found")
            continue

        data = mat_data['data']
        # 去除 baseline（假设使用前 3秒数据作为 baseline），同时只取前32个通道
        data_no_baseline = data[:, :32, n_baseline_points:]
        n_trials, n_channels, total_timepoints = data_no_baseline.shape

        all_features = []

        # 遍历每个 trial
        for t_idx in range(n_trials):
            trial_data = data_no_baseline[t_idx]  # shape: (n_channels, n_timepoints)
            trial_feature_list = []

            start = 0
            while start + win_size_points <= total_timepoints:
                # 取出当前窗口数据，shape: (n_channels, win_size_points)
                window_data = trial_data[:, start:start + win_size_points]
                # 初始化当前窗口的特征矩阵，shape: (n_channels, n_bands)
                window_de = np.zeros((n_channels, len(bands)))

                for ch_idx in range(n_channels):
                    signal_1s = window_data[ch_idx, :]
                    for b_idx, (b_name, b_low, b_high) in enumerate(bands):
                        # 带通滤波
                        sig_filt = bandpass_filter(signal_1s, b_low, b_high, fs)
                        # 计算差分熵
                        window_de[ch_idx, b_idx] = compute_de(sig_filt)

                trial_feature_list.append(window_de)  # 每个元素形状为 (n_channels, n_bands)
                start += step_points

            # 转换为 numpy 数组，形状: (n_segments, n_channels, n_bands)
            trial_feature_array = np.array(trial_feature_list)

            # 对每个通道、每个频段的时间序列进行 LDS 平滑
            n_segments, _, n_bands = trial_feature_array.shape
            smoothed_trial_features = np.empty_like(trial_feature_array)
            for ch in range(n_channels):
                for b in range(n_bands):
                    original_series = trial_feature_array[:, ch, b]
                    smoothed_series = lds_smoothing(original_series)
                    smoothed_trial_features[:, ch, b] = smoothed_series

            all_features.append(smoothed_trial_features)

        # 将所有 trial 的特征组合起来，形状: (n_trials, n_segments, n_channels, n_bands)
        all_features = np.array(all_features)

        # 保存处理后的数据到输出文件夹
        output_path = os.path.join(output_folder, f"preprocessed_{file_name}")
        sio.savemat(output_path, {'features': all_features})
        print(f"Saved: {output_path}")
