import logging
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
from nemo import lightning as nl

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)


def main(): 
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")
    asr_model.change_vocabulary(new_tokenizer_dir='tokenizer/',
                            new_tokenizer_type='bpe')
    
    freeze_encoder = True
    if freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
        
    train_config = {
    "manifest_filepath": "manifests/train_manifest.json",
    "val_manifest_filepath": "manifests/val_manifest.json",
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-4,
    "epochs": 10,
    "sample_rate": 16_000
}

    asr_model.setup_training_data(train_config)
    asr_model.setup_validation_data(train_config)
    
    # optimizer config
    with open_dict(asr_model.cfg.optim):
        asr_model.cfg.optim.lr = 0.01
        asr_model.cfg.optim.betas = [0.95, 0.5]  # from paper
        asr_model.cfg.optim.weight_decay = 0.001  # Original weight decay
        asr_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
        asr_model.cfg.optim.sched.warmup_ratio = 0.05  # 5 % warmup
        asr_model.cfg.optim.sched.min_lr = 1e-5
    trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=2,
    precision=32,
    )
    trainer.fit(asr_model)

    