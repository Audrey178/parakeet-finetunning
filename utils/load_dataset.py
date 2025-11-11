from glob import glob
import os
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import io
import json
import math
import argparse
from prepare_data import format_string
from datasets import Audio
from concurrent.futures import ThreadPoolExecutor


class LocalDataFormatter:
    def __init__(self, input_dir, output_dir):
        if input_dir is not None:
            self.input_dir = os.path.join(input_dir, 'data')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_data(self, cache_dir=None, batch_size=500, num_workers=4, keep_in_memory=False):
        '''
        Chuyển đổi dữ liệu từ file Parquet sang định dạng WAV và tạo manifest.
        Trả về danh sách các mẫu dữ liệu.
        '''
        input_dir = Path(self.input_dir)
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
    def save_manifest(self, data, filename):
        manifest_path = filename
        with open(manifest_path, "a", encoding="utf-8") as f:
            for sample in data:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")


class HuggingFaceDataFormatter(LocalDataFormatter):
    def __init__(self, input_dir, split, output_dir, manifest_dir, num_samples = 5000):
        self.dataset_name = input_dir
        self.manifest_dir = manifest_dir
        self.split = split
        self.num_samples = num_samples
        os.makedirs(manifest_dir, exist_ok=True)
        self.manifest_path = os.path.join(manifest_dir, f'{self.split}_manifest.json')
        super().__init__(input_dir=input_dir, output_dir=output_dir)
        
        
    def prepare_dataset(self, batch):
        audios = batch['audio']
        batch["array"] = [a["array"] for a in audios]
        batch['sampling_rate'] = [a["sampling_rate"] for a in audios]
        return batch
    
    def segment_dataset(self, dataset):
        print("Segment dataset...")
        
        for i in range(0, math.ceil(len(dataset)/ self.num_samples)):
            segment = dataset.select(list(range(i*self.num_samples, min((i + 1)*self.num_samples, len(dataset)))))
            yield segment.cast_column('audio', Audio(decode=True))

    def convert_data(self, cache_dir=None, batch_size=500, num_workers=4, keep_in_memory=False):
        from datasets import load_dataset
        
        dataset = load_dataset(self.dataset_name, split=self.split, keep_in_memory=keep_in_memory, cache_dir=cache_dir, streaming=keep_in_memory)
        len_samples = 0
        yield_segments = self.segment_dataset(dataset)
        yield_segments_2 = self.segment_dataset(dataset)
        
        seg_idx = [idx for idx, segment in enumerate(yield_segments_2)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for len_sample in ex.map(self.convert_segment, yield_segments, seg_idx):
                len_samples+=len_sample
                
        print(f'Tong so samples: {len_samples}')
        
    def convert_segment(self, segment, seg_idx):
        name = self.dataset_name.split('/')[-1]
        temp_datasets = segment
        seg_samples = []
        for idx, row in tqdm(enumerate(temp_datasets), total=len(temp_datasets), desc=f'Preprocessing segment {seg_idx}'):
            audio = row['audio']
            trans = format_string(row['transcription'])
                
            data = audio['array']
            sr = audio['sampling_rate']
                
            if data is not None:
                fname = f"hf_{name}_{seg_idx:03d}_{idx:06d}.wav"
                out_wav = os.path.join(self.output_dir, fname)
                sf.write(out_wav, data, sr)
            else:
                raise ValueError(f"Unexpected audio data: {data}")
                
            seg_samples.append({
                "audio_filepath": out_wav,
                "duration": len(data) / sr,
                "text": trans
                })
        self.save_manifest(seg_samples, self.manifest_path)
        return len(seg_samples)
                
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', help='Input directory containing the dataset', type=str, required=True)
    parser.add_argument('--output_dir', help='Output directory for processed data', type=str, required=True)
    parser.add_argument('--cache_dir', help='Cache directory for dataset', type=str, default=None)
    parser.add_argument('--batch_size', help='Batch size for processing', type=int, default=500)
    parser.add_argument('--num_workers', help='Number of worker processes', type=int, default=4)
    parser.add_argument('--keep_in_memory', help='Keep dataset in memory', action='store_true', default=False)
    parser.add_argument('--source', help='Source of the dataset', choices=['huggingface', 'local'], required=True)
    parser.add_argument('--split', help='Dataset split to use', choices=['train', 'val'], required=True)
    parser.add_argument('--num_samples', help='Number of samples per segment', type=int, default=5000)
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    split = str(args.split)
    output_dir = Path(args.output_dir) / split
    num_samples = args.num_samples
    cache_dir = args.cache_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    keep_in_memory = args.keep_in_memory
    source = args.source
    
    print("Processing dataset with the following parameters:")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {split}")
    manifest_dir = 'manifests'
    
    if source == 'huggingface':
        formatter = HuggingFaceDataFormatter(
            input_dir=input_dir,
            output_dir=output_dir,
            split=split,
            manifest_dir=manifest_dir,
            num_samples=num_samples
        )
    elif source == 'local':
        formatter = LocalDataFormatter(
            input_dir=input_dir,
            output_dir=output_dir,
        )
    
    all_result = formatter.convert_data(
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        keep_in_memory=keep_in_memory
    )
    