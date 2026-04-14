import os
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings

warnings.filterwarnings("ignore")

# 1. 设置路径与参数
data_path = '/Users/yanyuhan/Downloads/AudioWAV'
MAX_SEQ_LENGTH = 150  # 统一对齐到 150 帧 (约 3 秒音频)

emotion_map = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

print("正在加载 Wav2Vec 2.0 预训练模型...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()


# 2. 特征提取核心函数
def extract_temporal_features(file_path):
    # === A. 提取传统 45 维物理特征 (作为全局静态声学轮廓) ===
    audio_orig, sr_orig = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_orig, sr=sr_orig, n_mfcc=40)
    pitch = librosa.yin(audio_orig, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    zcr = librosa.feature.zero_crossing_rate(audio_orig)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_orig, sr=sr_orig)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_orig, sr=sr_orig)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_orig, sr=sr_orig)

    trad_features = np.concatenate((
        np.mean(mfccs.T, axis=0),
        [np.mean(pitch), np.mean(zcr), np.mean(spectral_centroid),
         np.mean(spectral_bandwidth), np.mean(spectral_rolloff)]
    ))  # 形状: (45,)

    # === B. 提取 Wav2Vec2 深度时序特征 (保留时间维度) ===
    audio_16k, _ = librosa.load(file_path, sr=16000)
    input_values = processor(audio_16k, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        outputs = wav2vec_model(input_values)
        hidden_state = outputs.last_hidden_state  # 形状: (1, seq_len, 768)

    # 核心：时序对齐 (Padding or Truncating)
    seq_len = hidden_state.shape[1]
    if seq_len < MAX_SEQ_LENGTH:
        # 如果音频太短，在时间维度(dim=1)补零
        pad_size = MAX_SEQ_LENGTH - seq_len
        # pad 格式: (最后一维左, 最后一维右, 倒数第二维左, 倒数第二维右)
        hidden_state = F.pad(hidden_state, (0, 0, 0, pad_size), "constant", 0)
    elif seq_len > MAX_SEQ_LENGTH:
        # 如果音频太长，直接截断
        hidden_state = hidden_state[:, :MAX_SEQ_LENGTH, :]

    deep_temporal_features = hidden_state.squeeze().numpy()  # 形状: (150, 768)

    return trad_features, deep_temporal_features


# 3. 遍历提取
if __name__ == "__main__":
    X_trad_list = []
    X_deep_list = []
    y_list = []
    speaker_id_list = []

    all_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    total_files = len(all_files)
    print(f"\n 开始提取时序特征,总计 {total_files} 个音频。")


    processed = 0

    for file_name in all_files:
        parts = file_name.split('_')
        if len(parts) >= 3:
            speaker_id = parts[0]  # 例如 '1001'
            emo_code = parts[2]  # 例如 'ANG'

            if emo_code in emotion_map:
                label = emotion_map[emo_code]
                file_path = os.path.join(data_path, file_name)

                try:
                    trad_feat, deep_feat = extract_temporal_features(file_path)

                    X_trad_list.append(trad_feat)
                    X_deep_list.append(deep_feat)
                    y_list.append(label)
                    speaker_id_list.append(speaker_id)

                    processed += 1
                    if processed % 100 == 0:
                        print(f"进度: 已处理 {processed}/{total_files} 个音频...")
                except Exception as e:
                    print(f"处理 {file_name} 时出错: {e}")

    # 4. 转换为 Numpy 数组并保存
    X_trad = np.array(X_trad_list)
    X_deep = np.array(X_deep_list)
    y_labels = np.array(y_list)
    speaker_ids = np.array(speaker_id_list)

    print("\n" + "=" * 50)
    print("特征提取圆满完成！")
    print(f"传统特征矩阵形状 (X_trad): {X_trad.shape}   # 预期 (7442, 45)")
    print(f"深度时序特征形状 (X_deep): {X_deep.shape} # 预期 (7442, 150, 768)")
    print(f"情感标签数量 (y_labels): {y_labels.shape}")
    print(f"说话人ID数量 (speaker_ids): {speaker_ids.shape}")
    print("=" * 50)

    # 分别保存为四个核心文件
    np.save('SCI_X_trad.npy', X_trad)
    np.save('SCI_X_deep_temporal.npy', X_deep)
    np.save('SCI_y_labels.npy', y_labels)
    np.save('SCI_speaker_ids.npy', speaker_ids)
    print(">> 已成功保存 4 个 SCI 级基础特征文件至本地目录")