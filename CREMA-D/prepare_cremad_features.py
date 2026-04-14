import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings

warnings.filterwarnings("ignore")

# 1. 设置 CREMA-D 的绝对路径 (根据你提供的信息)
data_path = '/Users/yanyuhan/Downloads/AudioWAV'

# 定义 CREMA-D 的情感标签映射字典
emotion_map = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# 2. 初始化预训练大模型
print("正在加载 Wav2Vec 2.0 预训练模型...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()


# 3. 提取 813 维双流特征
def extract_fused_features(file_path):
    # --- A. 提取传统 45 维特征 ---
    audio_orig, sr_orig = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_orig, sr=sr_orig, n_mfcc=40)
    pitch = librosa.yin(audio_orig, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    zcr = librosa.feature.zero_crossing_rate(audio_orig)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_orig, sr=sr_orig)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_orig, sr=sr_orig)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_orig, sr=sr_orig)

    mfccs_processed = np.mean(mfccs.T, axis=0)
    traditional_features = np.concatenate((
        mfccs_processed,
        [np.mean(pitch), np.mean(zcr), np.mean(spectral_centroid),
         np.mean(spectral_bandwidth), np.mean(spectral_rolloff)]
    ))

    # --- B. 提取 Wav2Vec 2.0 深度特征 (16000Hz) ---
    audio_16k, _ = librosa.load(file_path, sr=16000)
    input_values = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        last_hidden_state = outputs.last_hidden_state
        deep_features = last_hidden_state.mean(dim=1).squeeze().numpy()

        # --- C. 融合 ---
    return np.concatenate((traditional_features, deep_features))


# 4. 遍历提取
if __name__ == "__main__":
    X = []
    y = []

    # 统计一下总共有多少个 wav 文件
    all_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    total_files = len(all_files)
    print(f"\n检测到 {total_files} 个音频文件，开始提取特征（数据量较大，请耐心等待，预计需要半小时左右）...")

    processed = 0

    for file_name in all_files:
        # 解析 CREMA-D 文件名中的情感标签，例如 '1001_IEO_ANG_HI.wav' -> 'ANG'
        parts = file_name.split('_')
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in emotion_map:
                label = emotion_map[emo_code]
                file_path = os.path.join(data_path, file_name)

                try:
                    fused_feat = extract_fused_features(file_path)
                    X.append(fused_feat)
                    y.append(label)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"进度: 已处理 {processed}/{total_files} 个音频...")
                except Exception as e:
                    print(f"处理 {file_name} 时出错: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n特征提取完成！特征矩阵形状: {X.shape}")
    np.save('X_cremad_fused.npy', X)
    np.save('y_cremad_labels.npy', y)
    print("成功将融合特征保存为 X_cremad_fused.npy 和 y_cremad_labels.npy！")