import numpy as np


# 数据增强方法
def scale_frequency_bands(features, scale_range=(0.9, 1.1)):
    """
    随机缩放频域特征的特定频带。
    """
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=features.shape[-1])
    enhanced_features = features * scale_factors
    return enhanced_features


# 建议测试多个 frequency 和 amplitude 组合，如 (2, 0.05) 或 (5, 0.1)，以观察对性能的实际影响。
def add_periodic_perturbation(features, frequency=2, amplitude=0.05):
    """
    向频域特征添加正弦波扰动，增强周期性。
    """
    num_bands = features.shape[-1]
    time = np.linspace(0, 2 * np.pi, num_bands)
    sinusoid = amplitude * np.sin(frequency * time)
    enhanced_features = features + sinusoid
    return enhanced_features


def mixup(features, labels, alpha=0.2):
    """
    使用 Mixup 方法增强频域特征。
    """
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(features.shape[0])
    mixed_features = lam * features + (1 - lam) * features[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    return mixed_features, mixed_labels
