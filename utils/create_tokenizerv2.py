import subprocess
import os

BRANCH = 'r2.0.0rc0'

train_manifest_cleaned = 'manifests/train_manifest.json'
VOCAB_SIZE = 6000
tokenizer_dir = "tokenizer"
TOKENIZER_TYPE = 'bpe'

if __name__ == '__main__':
    if not os.path.exists("scripts/process_asr_text_tokenizer.py"):
        cmd = ["wget", "-P", "scripts", f"https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/scripts/tokenizers/process_asr_text_tokenizer.py"]
        subprocess.run(cmd)
    
    cmd = [
    "python", "scripts/process_asr_text_tokenizer.py",
    f"--manifest={train_manifest_cleaned}",
    f"--vocab_size={VOCAB_SIZE}",
    f"--data_root={tokenizer_dir}",
    "--tokenizer=spe",
    f"--spe_type={TOKENIZER_TYPE}",
    "--spe_character_coverage=1.0",
    "--no_lower_case",
    "--log"
    ]
    subprocess.run(cmd, check=True)