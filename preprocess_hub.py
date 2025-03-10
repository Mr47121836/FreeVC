import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)

def process(filename, feature_extractor, model, device):
    basename = os.path.basename(filename)
    speaker = basename[:7]  # 假设前 7 个字符为说话人 ID
    save_dir = os.path.join(args.out_dir, speaker)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载音频
    wav, _ = librosa.load(filename, sr=args.sr)
    # 预处理音频
    input_values = feature_extractor(wav, return_tensors="pt", sampling_rate=args.sr).input_values  # [1, T]
    input_values = input_values.to(device).half()  # 转换为半精度
    
    # 提取特征
    with torch.no_grad():
        outputs = model(input_values)
        c = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
    
    # 保存特征
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    torch.save(c.cpu(), save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./dataset/aishell-16k", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="./dataset/hubert/", help="path to output dir")
    parser.add_argument("--model_dir", type=str, default="./HuBERT/chinese-hubert-large-fariseq-ckpt", help="path to HuBERT model")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading HuBERT for content...")
    # 加载 HuBERT 模型和特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    cmodel = HubertModel.from_pretrained(args.model_dir)
    cmodel = cmodel.to(device)
    cmodel = cmodel.half()  # 使用半精度
    cmodel.eval()
    print("Success Loaded HuBERT.")

    # 遍历所有 WAV 文件
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    print(filenames)
    for filename in tqdm(filenames):
        process(filename, feature_extractor, cmodel, device)
