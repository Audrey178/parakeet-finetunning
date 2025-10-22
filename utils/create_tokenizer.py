import pandas as pd
manifests = ['manifests/train_manifest.json', 'manifests/val_manifest.json']
import sentencepiece as spm

train_text_path = 'train_text.txt'

if __name__ == "__main__":
  for manifest in manifests:
    ds = pd.read_json(manifest, lines=True)
    for text in ds['text']:
      with open(train_text_path, 'a', encoding='utf-8') as f:
        f.write(f"{text}\n")
  
  tokenize = spm.SentencePieceTrainer.train(f'--input={train_text_path} --model_prefix=tokenizer --vocab_size=6000')
  input_vocab = "tokenizer.vocab"
  output_vocab = "vocab.txt"

  with open(input_vocab, "r", encoding="utf-8") as fin, open(output_vocab, "w", encoding="utf-8") as fout:
    for line in fin:
        token = line.strip().split("\t")[0].split(" ")[0]  # tách token
        if token:  # bỏ dòng trống
            fout.write(token + "\n")

  print(f"✅ Created vocab.txt with {sum(1 for _ in open(output_vocab))} tokens at {output_vocab}")
