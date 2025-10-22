from glob import glob
import os
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import io
import json
SAMPLING_RATE = 16_000
input_dir= 'datasets/vlsp2020_vinai_100h/data'
output_dir= 'data'
manifest_dir = 'manifests'

def convert_data(input_dir, output_dir):
    input_dir = Path("datasets") / "vlsp2020_vinai_100h" / "data"
    input_paths = list(input_dir.glob("*.parquet"))
    all_samples = []
    for idx_path, path in enumerate(input_paths):
        ds = pd.read_parquet(path)
        for idx, row in tqdm(ds.iterrows(), total=len(ds), desc=f'Preprocessing {path}'):
            audio = row['audio']
            trans = row['transcription']

            # === ĐỌC DỮ LIỆU AUDIO TỪ BYTES ===
            if isinstance(audio, dict) and 'bytes' in audio:
                audio_bytes = io.BytesIO(audio['bytes'])
                data, samplerate = sf.read(audio_bytes)
            else:
                raise ValueError(f"Unexpected audio format: {type(audio)}")

            # === GHI LẠI THÀNH FILE WAV ===
            fname = f"{idx_path:03d}_{idx:06d}.wav"
            out_wav = os.path.join(output_dir, fname)
            sf.write(out_wav, data, samplerate)

            # === LƯU THÔNG TIN VÀO DANH SÁCH ===
            all_samples.append({
                "audio_filepath": out_wav,
                "duration": len(data) / samplerate,
                "text": trans.strip().lower(),
            })
    print(f"Tổng số mẫu: {len(all_samples)}")
    return all_samples



# --- Ghi ra file manifest JSON ---
def save_manifest(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    all_samples = convert_data(input_dir, output_dir)
    train_ratio = 0.95
    train_samples = all_samples[: int(len(all_samples) * train_ratio)]
    val_samples   = all_samples[int(len(all_samples) * train_ratio):]
    os.makedirs(manifest_dir, exist_ok=True)
    save_manifest(train_samples, os.path.join(manifest_dir, "train_manifest.json"))
    save_manifest(val_samples,   os.path.join(manifest_dir, "val_manifest.json"))
