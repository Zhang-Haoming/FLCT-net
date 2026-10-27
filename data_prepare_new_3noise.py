import math

import numpy as np
# Author: Haoming Zhang
#The code here not only include data importing, but also data standardization and the generation of analog noise signals

def get_rms(records):
   
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def get_rms_vectorized(records):
    """
    向量化计算均方根（RMS），支持二维数组（每行一个样本）
    :param records: 输入数组，shape=(n_samples, n_points)
    :return: 每行的RMS值，shape=(n_samples, 1)
    """
    # 向量化计算：平方 → 按行求均值 → 开平方 → 保持二维方便广播
    rms_vals = np.sqrt(np.mean(np.square(records), axis=1, keepdims=True))
    return rms_vals

def _scale_noise_to_snr(eeg, noise, snr_db):
    """Scale each noise segment to the requested RMS-based SNR."""
    snr_ratio = 10 ** (np.asarray(snr_db) / 20)
    if snr_ratio.ndim == 1:
        snr_ratio = snr_ratio.reshape(-1, 1)

    eeg_rms = get_rms_vectorized(eeg)
    noise_rms = get_rms_vectorized(noise)
    return noise * eeg_rms / (noise_rms * snr_ratio)


NOISE_COMBINATIONS = {
    "EOG": ("EOG",),
    "EMG": ("EMG",),
    "ECG": ("ECG",),
    "EOG+EMG": ("EOG", "EMG"),
    "EOG+ECG": ("EOG", "ECG"),
    "EMG+ECG": ("EMG", "ECG"),
    "ALL": ("EOG", "EMG", "ECG"),
}


def _normalize_noise_type(noise_type):
    """Return one canonical name for the seven supported noise settings."""
    value = str(noise_type).upper().replace(" ", "")
    if value == "ALL":
        return "ALL"

    parts = value.split("+")
    if not parts or any(part not in {"EOG", "EMG", "ECG"} for part in parts):
        raise ValueError(
            "Unsupported noise_type. Use EOG, EMG, ECG, EOG+EMG, "
            "EOG+ECG, EMG+ECG, or all."
        )
    if len(parts) != len(set(parts)):
        raise ValueError(f"Repeated artifact in noise_type: {noise_type}")

    part_set = set(parts)
    for canonical_name, canonical_parts in NOISE_COMBINATIONS.items():
        if part_set == set(canonical_parts):
            return canonical_name
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def _build_noise_for_condition(eeg, noise_data, noise_type, snr_db):
    """Build one artifact condition and enforce its final RMS-based SNR.

    For a combination, every component is first scaled to the same SNR
    relative to EEG. The components are then summed, and the summed artifact
    is scaled once more so that its total SNR relative to EEG equals snr_db.
    """
    canonical_type = _normalize_noise_type(noise_type)
    artifact_names = NOISE_COMBINATIONS[canonical_type]

    if len(artifact_names) == 1:
        return _scale_noise_to_snr(
            eeg, noise_data[artifact_names[0]], snr_db
        )

    balanced_components = [
        _scale_noise_to_snr(eeg, noise_data[name], snr_db)
        for name in artifact_names
    ]
    combined_noise = np.sum(np.stack(balanced_components, axis=0), axis=0)
    return _scale_noise_to_snr(eeg, combined_noise, snr_db)

def random_signal(signal,combin_num):
    # Random disturb and augment signal
    random_result=[]

    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
        random_result.append(shuffled_dataset)
    
    random_result  = np.array(random_result)

    return  random_result
        

def split_dataset(data, train_per):
    """
    通用数据集划分函数：按比例划分训练/验证/测试集
    :param data: 随机化后的单类数据，shape=(样本数, 时间点)
    :param train_per: 训练集比例
    :return: 训练集、验证集、测试集
    """
    total_num = data.shape[0]
    train_num = round(train_per * total_num)
    validation_num = round((total_num - train_num) / 2)
    test_num = total_num - train_num - validation_num

    # 划分数据集（互不重叠）
    train_data = data[0:train_num, :]
    val_data = data[train_num:train_num+validation_num, :]
    test_data = data[train_num+validation_num:total_num, :]
    
    # print(f"数据总量: {total_num} | 训练集: {train_data.shape[0]} | 验证集: {val_data.shape[0]} | 测试集: {test_data.shape[0]}")
    return train_data, val_data, test_data

def simple_four_signal_pairs(eeg_data, eog_data, emg_data, ecg_data, multiple=10):
    """
    EEG与三类伪迹的随机配对算法：
    1. 以EEG样本数为基准，对EOG/EMG/ECG做“多删少补”
    2. 随机配对EEG/EOG/EMG/ECG，重复multiple倍次数
    3. 最终返回trial*timepoints维度的配对数据
    
    :param eeg_data: EEG数据，shape=(eeg_num, time_points)
    :param eog_data: EOG数据，shape=(eog_num, time_points)
    :param emg_data: EMG数据，shape=(emg_num, time_points)
    :param ecg_data: ECG数据，shape=(ecg_num, time_points)
    :param multiple: 最终配对总数 = EEG样本数 * multiple（默认10）
    :return:
        paired_eeg: 配对后EEG数据，shape=(total_trials, time_points)
        paired_eog: 配对后EOG数据，shape=(total_trials, time_points)
        paired_emg: 配对后EMG数据，shape=(total_trials, time_points)
        paired_ecg: 配对后ECG数据，shape=(total_trials, time_points)
    """
    # 1. 基础参数
    eeg_num, time_points = eeg_data.shape  # EEG样本数 + 时间点维度
    target_num_per_round = eeg_num         # 每一轮的目标样本数（和EEG一致）
    total_trials = eeg_num * multiple      # 最终总配对数

    # 2. 对EOG做“多删少补”，使其样本数=EEG样本数
    def adjust_data(data, target_num):
        """辅助函数：调整数据样本数到目标值（多删少补）"""
        data_num = data.shape[0]
        if data_num == target_num:
            return data  # 数量一致，直接返回
        elif data_num > target_num:
            # 数量多：随机筛选出target_num个样本
            idx = np.random.choice(data_num, target_num, replace=False)
            return data[idx]
        else:
            # 数量少：随机复用，直到达到target_num个样本
            repeat_times = target_num // data_num + 1
            data_repeated = np.tile(data, (repeat_times, 1))  # 重复拼接
            idx = np.random.choice(data_repeated.shape[0], target_num, replace=False)  # 随机选，避免连续重复
            return data_repeated[idx]

    # 调整三类伪迹样本数到和EEG一致
    eog_adjusted = adjust_data(eog_data, target_num_per_round)
    emg_adjusted = adjust_data(emg_data, target_num_per_round)
    ecg_adjusted = adjust_data(ecg_data, target_num_per_round)

    # 3. 多次随机配对（重复multiple次）
    paired_eeg_list = []
    paired_eog_list = []
    paired_emg_list = []
    paired_ecg_list = []

    for _ in range(multiple):

        # 调整EOG、EMG和ECG样本数到和EEG一致
        eog_adjusted = adjust_data(eog_data, target_num_per_round)
        emg_adjusted = adjust_data(emg_data, target_num_per_round)
        ecg_adjusted = adjust_data(ecg_data, target_num_per_round)

        # 每一轮独立打乱，随后按行一一配对
        eeg_shuffled = eeg_data[np.random.permutation(eeg_num)]
        eog_shuffled = eog_adjusted[np.random.permutation(target_num_per_round)]
        emg_shuffled = emg_adjusted[np.random.permutation(target_num_per_round)]
        ecg_shuffled = ecg_adjusted[np.random.permutation(target_num_per_round)]
        
        # 本轮配对（按打乱后的顺序一一对应）
        paired_eeg_list.append(eeg_shuffled)
        paired_eog_list.append(eog_shuffled)
        paired_emg_list.append(emg_shuffled)
        paired_ecg_list.append(ecg_shuffled)

    # 4. 拼接所有轮次的数据，整理为trial*timepoints维度
    paired_eeg = np.concatenate(paired_eeg_list, axis=0)[:total_trials]  # 截断到目标总数（防止误差）
    paired_eog = np.concatenate(paired_eog_list, axis=0)[:total_trials]
    paired_emg = np.concatenate(paired_emg_list, axis=0)[:total_trials]
    paired_ecg = np.concatenate(paired_ecg_list, axis=0)[:total_trials]

    # 打印日志
    print("=== EEG与三类伪迹配对结果 ===")
    print(f"原始EEG样本数: {eeg_num}, EOG样本数: {eog_data.shape[0]}, EMG样本数: {emg_data.shape[0]}, ECG样本数: {ecg_data.shape[0]}")
    print(f"调整后EOG样本数: {eog_adjusted.shape[0]}, EMG样本数: {emg_adjusted.shape[0]}, ECG样本数: {ecg_adjusted.shape[0]}")
    print(f"最终总配对数: {paired_eeg.shape[0]} (EEG数*{multiple} = {eeg_num}*{multiple})")
    print(f"配对后EEG shape: {paired_eeg.shape} (trial*timepoints)")
    print(f"配对后EOG shape: {paired_eog.shape} (trial*timepoints)")
    print(f"配对后EMG shape: {paired_emg.shape} (trial*timepoints)")
    print(f"配对后ECG shape: {paired_ecg.shape} (trial*timepoints)")

    return paired_eeg, paired_eog, paired_emg, paired_ecg

# 主函数
def prepare_data_new(EEG_all, EOG_noise_all, EMG_noise_all, ECG_noise_all, train_combin_num, val_combin_num, train_per, noise_type, train_db=[-5,5], test_db=[-5,5,10], clean_EEG=1, val_clean_EEG=0):
    # Here we use eeg and noise signal to generate scale transed training, validation, test signal

    # 随机打乱数据
    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    EOG_noise_random = np.squeeze(random_signal(signal=EOG_noise_all, combin_num=1))
    EMG_noise_random = np.squeeze(random_signal(signal=EMG_noise_all, combin_num=1))
    ECG_noise_random = np.squeeze(random_signal(signal=ECG_noise_all, combin_num=1))
    noise_type = _normalize_noise_type(noise_type)

    # 划分EEG数据集
    train_eeg, validation_eeg, test_eeg = split_dataset(EEG_all_random, train_per)

    # 划分EOG数据集
    train_eog, validation_eog, test_eog = split_dataset(EOG_noise_random, train_per)

    # 划分EMG数据集
    train_emg, validation_emg, test_emg = split_dataset(EMG_noise_random, train_per)

    # 划分ECG数据集
    train_ecg, validation_ecg, test_ecg = split_dataset(ECG_noise_random, train_per)


    #################################  simulate noise signal of training set  ##############################

    # -------------------------- 第一步：生成四种信号的配对数据 --------------------------
    EEG_paired, EOG_paired, EMG_paired, ECG_paired = simple_four_signal_pairs(
        eeg_data=train_eeg,
        eog_data=train_eog,
        emg_data=train_emg,
        ecg_data=train_ecg,
        multiple=train_combin_num,
    )
    
    # -------------------------- 第二步：叠加指定类型噪声 --------------------------

    trial_num = EEG_paired.shape[0]

    snr_min, snr_max = train_db

    SNR_train_dB = np.random.uniform(snr_min, snr_max, trial_num)
    eeg = EEG_paired
    noise_data = {
        "EOG": EOG_paired,
        "EMG": EMG_paired,
        "ECG": ECG_paired,
    }
    noise_adjust = _build_noise_for_condition(
        eeg, noise_data, noise_type, SNR_train_dB
    )
    noiseEEG_train = eeg + noise_adjust


    # -------------------------- 第三步：先追加干净EEG，再统一标准化 --------------------------
    clean_eeg_num = 0  # 默认不追加
    if clean_EEG > 0:
        # 计算追加数量：clean_EEG × 原始训练集EEG数量
        clean_eeg_num = train_eeg.shape[0] * clean_EEG
        
        # 随机采样干净EEG（未标准化）
        clean_eeg_idx = np.random.choice(EEG_paired.shape[0], clean_eeg_num, replace=True)
        clean_eeg = EEG_paired[clean_eeg_idx]
        
        # 拼接：含噪EEG + 干净EEG、原始EEG + 干净EEG（均为未标准化数据）
        noiseEEG_train_all = np.concatenate([noiseEEG_train, clean_eeg], axis=0)
        EEG_paired_all = np.concatenate([EEG_paired, clean_eeg], axis=0)
        
        print(f"追加干净EEG | 倍数: {clean_EEG} | 数量: {clean_eeg_num} | 拼接后总数量: {noiseEEG_train_all.shape[0]}")
    else:
        # 不追加干净EEG，直接使用原数据
        noiseEEG_train_all = noiseEEG_train
        EEG_paired_all = EEG_paired

    # -------------------------- 统一标准化（向量化优化版） --------------------------
    # 1. 计算拼接后所有含噪EEG样本的标准差（含干净EEG，干净EEG的标准差=自身标准差）
    std_vals = np.std(noiseEEG_train_all, axis=1, keepdims=True)  # shape=(total_num, 1)

    # 2. 对所有数据统一标准化（含噪EEG+干净EEG用同一套标准差）
    EEG_standard = EEG_paired_all / std_vals
    noiseEEG_standard = noiseEEG_train_all / std_vals

    # -------------------------- 第五步：输出日志 --------------------------
    print(f"\n数据生成完成 | 噪声类型: {noise_type} | 追加干净EEG倍数: {clean_EEG}")
    print(f"含噪EEG shape: {noiseEEG_standard.shape} | 原始EEG shape: {EEG_standard.shape}")

    # 3. 处理验证集
    noiseEEG_val_standard, EEG_val_standard, _ = process_val_test_set(
        eeg_data=validation_eeg,
        eog_data=validation_eog,
        emg_data=validation_emg,
        ecg_data=validation_ecg,
        val_test_type="验证",
        noise_type=noise_type,
        test_db=test_db,
        val_combin_num=val_combin_num,
        val_clean_EEG=val_clean_EEG
    )

    # 4. 处理测试集
    noiseEEG_test_standard, EEG_test_standard, std_VALUE_test = process_val_test_set(
        eeg_data=test_eeg,
        eog_data=test_eog,
        emg_data=test_emg,
        ecg_data=test_ecg,
        val_test_type="测试",
        noise_type=noise_type,
        test_db=test_db,
        val_combin_num=val_combin_num,
        val_clean_EEG=val_clean_EEG
    )

    # 5. 输出最终日志
    print(f"\n===== 所有数据处理完成 =====")
    print(f"训练集 | 含噪EEG shape: {noiseEEG_standard.shape}")
    print(f"验证集 | 含噪EEG shape: {noiseEEG_val_standard.shape}")
    print(f"测试集 | 含噪EEG shape: {noiseEEG_test_standard.shape}")


    return noiseEEG_standard, EEG_standard, noiseEEG_val_standard, EEG_val_standard, noiseEEG_test_standard, EEG_test_standard, std_VALUE_test


# -------------------------- 验证集/测试集处理核心函数 --------------------------
def process_val_test_set(eeg_data, eog_data, emg_data, ecg_data, val_test_type, noise_type, test_db, val_combin_num=1,val_clean_EEG=0):
    """
    处理验证集/测试集：固定SNR生成噪声数据 + 追加干净EEG
    :param eeg_data: 验证/测试集EEG（val_eeg/test_eeg）
    :param eog_data: 验证/测试集EOG
    :param emg_data: 验证/测试集EMG
    :param ecg_data: 验证/测试集ECG
    :param val_test_type: 类型标识（"validation"/"test"）
    :param noise_type: 七种噪声条件之一
    :param test_db: SNR范围 [min, max]
    :param val_combin_num: 配对倍数（默认1）
    :return: 标准化后的EEG、含噪EEG、标准差、SNR列表
    """
    # 第一步：生成四种信号的配对数据
    EEG_paired, EOG_paired, EMG_paired, ECG_paired = simple_four_signal_pairs(
        eeg_data=eeg_data,
        eog_data=eog_data,
        emg_data=emg_data,
        ecg_data=ecg_data,
        multiple=val_combin_num,
    )
    trial_num = EEG_paired.shape[0]
    noise_type = _normalize_noise_type(noise_type)
    noise_data = {
        "EOG": EOG_paired,
        "EMG": EMG_paired,
        "ECG": ECG_paired,
    }

    # 第二步：生成固定SNR值（test_db[2]个等间距值）
    SNR_test_dB = np.linspace(test_db[0], test_db[1], num=test_db[2]) # 会生成从test_db[0]到 test_db[1]的test_db[2]个数
    print(f"\n{val_test_type}集 | 固定SNR(dB): {np.round(SNR_test_dB, 2)}")

    # 第三步：按固定SNR生成噪声数据（向量化优化）
    # 初始化存储变量
    all_eeg = []       # 原始EEG（所有SNR）
    all_noise_eeg = [] # 含噪EEG（所有SNR）
    
    for snr_db in SNR_test_dB:
        eeg = EEG_paired
        noise_adjust = _build_noise_for_condition(
            eeg, noise_data, noise_type, snr_db
        )
        noise_eeg = eeg + noise_adjust

        # 追加当前SNR的结果
        all_eeg.append(eeg)
        all_noise_eeg.append(noise_eeg)

    # 拼接所有SNR的结果（shape=(10*trial_num, time_points)）
    EEG_test = np.concatenate(all_eeg, axis=0)
    noise_EEG_test = np.concatenate(all_noise_eeg, axis=0)

    # 第四步：追加无噪声的干净EEG数据
    # 采样和验证/测试集同等数量的干净EEG
    if val_clean_EEG > 0:
        clean_eeg_num = trial_num  # 与单SNR样本数一致
        clean_eeg_idx = np.random.choice(EEG_paired.shape[0], clean_eeg_num, replace=True)
        clean_eeg = EEG_paired[clean_eeg_idx]
        
        # 拼接干净EEG到末尾
        EEG_test = np.concatenate([EEG_test, clean_eeg], axis=0)
        noise_EEG_test = np.concatenate([noise_EEG_test, clean_eeg], axis=0)  # 干净EEG的含噪版=自身
        print(f"{val_test_type}集 | 追加干净EEG数量: {clean_eeg_num} | 最终总样本数: {EEG_test.shape[0]}")

    # 第五步：标准化（向量化+保存标准差）
    # 计算每个样本的标准差
    std_VALUE = np.std(noise_EEG_test, axis=1, keepdims=True)
    # 向量化标准化
    EEG_test_standard = EEG_test / std_VALUE
    noiseEEG_test_standard = noise_EEG_test / std_VALUE

    # 返回结果
    return noiseEEG_test_standard, EEG_test_standard, std_VALUE
  


# -------------------------- 轻量级SNR自检 --------------------------
def test_noise_combinations():
    """Check that all seven conditions reach the requested total SNR."""
    rng = np.random.default_rng(1234)
    eeg = rng.normal(size=(8, 1024))
    noise_data = {
        "EOG": rng.normal(size=(8, 1024)),
        "EMG": rng.normal(size=(8, 1024)),
        "ECG": rng.normal(size=(8, 1024)),
    }
    target_snr_db = -3.0

    for noise_type in NOISE_COMBINATIONS:
        noise = _build_noise_for_condition(
            eeg, noise_data, noise_type, target_snr_db
        )
        achieved_snr = 20 * np.log10(
            get_rms_vectorized(eeg) / get_rms_vectorized(noise)
        )
        if not np.allclose(achieved_snr, target_snr_db, atol=1e-10):
            raise AssertionError(
                f"SNR check failed for {noise_type}: {achieved_snr.ravel()}"
            )
        print(f"{noise_type}: target and achieved SNR = {target_snr_db:.1f} dB")


if __name__ == "__main__":
    test_noise_combinations()
