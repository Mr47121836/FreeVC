import os
import argparse
import torch
import librosa
import json
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
import multiprocessing as mp

import utils
from mel_processing import mel_spectrogram_torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
import logging
logging.getLogger("numba").setLevel(logging.WARNING)

# Global variables to share across processes
global_args = None
global_hps = None
global_feature_extractor = None
global_cmodel = None
global_vocoder = None
global_device = None

def init_worker(args, hps):
    """Initialize worker processes with shared global variables and load models."""
    global global_args, global_hps, global_feature_extractor, global_cmodel, global_vocoder, global_device
    global_args = args
    global_hps = hps
    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load HuBERT model and feature extractor in each worker
    print(f"Worker {mp.current_process().name}: Loading HuBERT for content...")
    global_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    global_cmodel = HubertModel.from_pretrained(args.model_dir)
    global_cmodel = global_cmodel.to(global_device)
    global_cmodel = global_cmodel.half()  # Use half precision
    global_cmodel.eval()
    print(f"Worker {mp.current_process().name}: Success Loaded HuBERT.")

    # Load vocoder in each worker
    print(f"Worker {mp.current_process().name}: Loading vocoder...")
    global_vocoder = utils.get_vocoder(0)
    global_vocoder.eval()
    print(f"Worker {mp.current_process().name}: Loaded vocoder.")

    print(f"Worker {mp.current_process().name} initialized with device: {global_device}, CUDA available: {torch.cuda.is_available()}")

def process(filename):
    """Process a single audio file."""
    if not filename.endswith('.wav'):
        return
    
    full_path = filename if os.path.isabs(filename) else os.path.join(global_args.in_dir, filename.split('/')[0], filename.split('/')[-1])
    basename = os.path.basename(full_path)
    speaker = full_path.split("/")[-2]  # Use directory name as speaker ID
    wav_dir = os.path.join(global_args.wav_dir, speaker)
    ssl_dir = os.path.join(global_args.ssl_dir, speaker)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(ssl_dir, exist_ok=True)
    
    # Load audio
    wav, _ = librosa.load(full_path, sr=global_hps.sampling_rate)
    wav = torch.from_numpy(wav).unsqueeze(0).to(global_device)
    
    # Compute mel spectrogram
    mel = mel_spectrogram_torch(
        wav,
        global_hps.n_fft,
        global_hps.num_mels,
        global_hps.sampling_rate,
        global_hps.hop_size,
        global_hps.win_size,
        global_hps.fmin,
        global_hps.fmax
    )
    
    # Process different frequency ranges
    for i in range(global_args.min, global_args.max + 1):
        mel_rs = utils.transform(mel, i)
        wav_rs = global_vocoder(mel_rs)[0][0].detach().cpu().numpy()
        _wav_rs = librosa.resample(wav_rs, orig_sr=global_hps.sampling_rate, target_sr=global_args.sr)
        wav_rs = torch.from_numpy(_wav_rs).to(global_device).unsqueeze(0)
        
        # Extract HuBERT features
        input_values = global_feature_extractor(wav_rs.squeeze(0).cpu().numpy(), return_tensors="pt", sampling_rate=global_args.sr).input_values
        input_values = input_values.to(global_device).half()
        
        with torch.no_grad():
            outputs = global_cmodel(input_values)
            c = outputs.last_hidden_state  # [1, seq_len, hidden_size]
            c = c.transpose(1, 2)  # Convert to [hidden_size, seq_len] for WavLM compatibility
        
        # Save SSL features
        ssl_path = os.path.join(ssl_dir, basename.replace(".wav", f"_{i}.pt"))
        torch.save(c.cpu(), ssl_path)
        
        # Save resampled audio
        wav_path = os.path.join(wav_dir, basename.replace(".wav", f"_{i}.wav"))
        wavfile.write(wav_path, global_args.sr, _wav_rs)

if __name__ == "__main__":
    # Add a version marker to confirm this is the updated code
    print("Running updated version with spawn method and worker-loaded models (2025-03-10 fix v2)")

    # Set multiprocessing start method to 'spawn' at the very beginning
    print("Setting multiprocessing start method to 'spawn'")
    mp.set_start_method('spawn', force=True)  # force=True ensures it overrides any previous setting

    # Verify the start method
    print(f"Current multiprocessing start method: {mp.get_start_method()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--min", type=int, default=68, help="min")
    parser.add_argument("--max", type=int, default=92, help="max")
    parser.add_argument("--config", type=str, default="hifigan/config.json", help="path to config file")
    parser.add_argument("--in_dir", type=str, default="./dataset/aishell-16k", help="path to input dir")
    parser.add_argument("--wav_dir", type=str, default="./dataset/sr/wav", help="path to output wav dir")
    parser.add_argument("--ssl_dir", type=str, default="./dataset/sr/hubert", help="path to output ssl dir")
    parser.add_argument("--model_dir", type=str, default="./HuBERT/chinese-hubert-large-fariseq-ckpt", help="path to HuBERT model")
    parser.add_argument("--num_workers", type=int, default=4, help="number of worker processes")
    args = parser.parse_args()

    # Set device in main process for reference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main process device: {device}, CUDA available: {torch.cuda.is_available()}")

    # Load config only in main process
    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)

    # Get all WAV files
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    filenames = [f.replace(args.in_dir + '/', '') for f in filenames]  # Relative paths for simplicity

    # Process files using multiprocessing
    with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args, hps)) as pool:
        for speaker in os.listdir(args.in_dir):
            spk_dir = os.path.join(args.in_dir, speaker)
            if os.path.isdir(spk_dir):
                # Filter files for this speaker
                speaker_files = [f for f in filenames if f.startswith(speaker + '/')]
                if speaker_files:
                    print(f"Processing speaker: {speaker}")
                    for _ in tqdm(pool.imap_unordered(process, speaker_files), total=len(speaker_files)):
                        pass

    print("Processing complete.")
