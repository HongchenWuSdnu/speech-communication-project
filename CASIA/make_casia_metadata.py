import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/root/autodl-tmp/CASIA/raw")
OUT  = Path("/root/autodl-tmp/CASIA/metadata.csv")
BAD  = Path("/root/autodl-tmp/CASIA/metadata_unparsed.csv")

# ======= 你论文要用的“统一情感标签” =======
# CASIA 常见 6 类：angry/fear/happy/neutral/sad/surprise
# 下面做了很多同义词/缩写/中英混写的兼容（不认识就算解析失败）
EMO_SYNONYMS = {
    "Anger": [
        "anger","angry","ang","an","a",
        "愤怒","生气"
    ],
    "Fear": [
        "fear","afraid","fea","f",
        "害怕","恐惧"
    ],
    "Happiness": [
        "happy","happiness","hap","h","joy","j",
        "高兴","开心","喜悦"
    ],
    "Neutral": [
        "neutral","neu","n","normal",
        "中性","平静","普通"
    ],
    "Sadness": [
        "sad","sadness","sadn","sa","s",
        "悲伤","难过"
    ],
    "Surprise": [
        "surprise","surprised","sur","su",
        "惊讶"
    ],
}

# 反向索引：token -> canonical label
TOKEN2EMO = {}
for canon, toks in EMO_SYNONYMS.items():
    for t in toks:
        TOKEN2EMO[t.lower()] = canon

# ======= 解析 emotion：先看路径片段，再看文件名 token =======
def parse_emotion(path: Path) -> str:
    s = str(path).lower()

    # 1) 优先：目录名 / 文件名里出现明确情感 token
    #    例如 .../angry/... 或 ..._ang_... 或 angry.wav
    for token, canon in TOKEN2EMO.items():
        if f"/{token}/" in s or f"_{token}_" in s or s.endswith(f"_{token}.wav") or s.endswith(f"_{token}.wav".upper()):
            return canon

    # 2) 再试：把路径和文件名按非字母数字切成 tokens
    parts = re.split(r"[^a-z0-9]+", s)
    for p in parts:
        if p in TOKEN2EMO:
            return TOKEN2EMO[p]

    return ""

# ======= 解析 speaker：尽量稳健，常见模式都抓 =======
# 你只需要 speaker_id 能把同一说话人归到一组即可（不用是“真实姓名”）
SPEAKER_PATTERNS = [
    r"(spk\d+)",
    r"(speaker\d+)",
    r"(s\d{1,2})",
    r"(m\d{1,3})",
    r"(f\d{1,3})",
    r"(p\d{1,3})",
    r"(id\d{1,4})",
]

def parse_speaker(path: Path) -> str:
    # 你的数据形如: ZhaoZuoxiang_201.wav
    # speaker 就取 "_" 前面这一段
    name = path.stem
    if name.startswith("._"):
        name = name[2:]  # 去掉 AppleDouble 前缀
    if "_" in name:
        spk = name.split("_", 1)[0].strip()
        if spk:
            return spk.lower()

    # 兜底：仍保留原先的正则策略（以防有少量不同命名）
    s = str(path).lower()
    for pat in SPEAKER_PATTERNS:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    m = re.match(r"([a-z]+\d{1,4})", path.stem.lower())
    if m:
        cand = m.group(1)
        if cand not in TOKEN2EMO:
            return cand

    return ""

def main():
    wavs = sorted([p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"])
    if not wavs:
        # 兼容 .WAV
        wavs = sorted([p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".wav".lower()])

    rows_ok = []
    rows_bad = []

    emo_cnt = Counter()
    spk_cnt = Counter()
    bad_reason = Counter()
    bad_examples = defaultdict(list)

    for wav in wavs:
        # ✅ 跳过 macOS AppleDouble 伪文件（资源分叉）
        if wav.name.startswith("._"):
            continue

        emo = parse_emotion(wav)
        spk = parse_speaker(wav)

        if emo and spk:
            rows_ok.append((str(wav), emo, spk))
            emo_cnt[emo] += 1
            spk_cnt[spk] += 1
        else:
            reason = []
            if not emo: reason.append("no_emotion")
            if not spk: reason.append("no_speaker")
            r = "+".join(reason)
            rows_bad.append((str(wav), "", "", r))
            bad_reason[r] += 1
            if len(bad_examples[r]) < 10:
                bad_examples[r].append(str(wav))

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wav_path", "emotion", "speaker"])
        w.writerows(rows_ok)

    with BAD.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wav_path", "emotion", "speaker", "reason"])
        w.writerows(rows_bad)

    print("======== CASIA metadata generation ========")
    print("ROOT:", ROOT)
    print("Total wav:", len(wavs))
    print("Parsed OK:", len(rows_ok))
    print("Unparsed :", len(rows_bad))
    print("\n[Emotion distribution]")
    for k, v in emo_cnt.most_common():
        print(f"  {k:10s}: {v}")

    print("\n[Speaker count]")
    print("  unique speakers:", len(spk_cnt))
    for k, v in spk_cnt.most_common(10):
        print(f"  {k:10s}: {v}")

    if rows_bad:
        print("\n[Unparsed reasons]")
        for k, v in bad_reason.most_common():
            print(f"  {k:18s}: {v}")
        print("\n[Examples of unparsed]")
        for k, exs in bad_examples.items():
            print(f"  - {k}:")
            for e in exs:
                print("     ", e)

    print("\nSaved:")
    print("  ", OUT)
    print("  ", BAD)

if __name__ == "__main__":
    main()