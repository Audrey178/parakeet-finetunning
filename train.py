import os
import copy
import logging
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn
import pytorch_lightning as ptl
import nemo.collections.asr as nemo_asr

# --- Configs / constants ---
LANGUAGE = 'vi'
TOKENIZER_DIR = 'tokenizer/tokenizer_spe_bpe_v6000'
TRAIN_MANIFEST = "manifests/train_manifest.json"
VAL_MANIFEST = "manifests/val_manifest.json"
EPOCHS = 10

logging.basicConfig(level=logging.INFO)


def enable_bn_se(m):
    # Keep BatchNorm1d and SqueezeExcite trainable (useful when encoder frozen)
    if isinstance(m, nn.BatchNorm1d):
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)


def main():
    # --- Load pretrained model ---
    logging.info("Loading pre-trained RNNT model from NeMo hub...")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-rnnt-1.1b"
    )

    # --- Optional: check tokenizer dir exists ---
    if not os.path.isdir(TOKENIZER_DIR):
        logging.warning(f"TOKENIZER_DIR '{TOKENIZER_DIR}' does not exist. "
                        "change_vocabulary may fail if tokenizer files are missing.")

    # --- Change vocabulary / tokenizer ---
    try:
        asr_model.change_vocabulary(new_tokenizer_dir=TOKENIZER_DIR,
                                    new_tokenizer_type='bpe')
        logging.info("Vocabulary changed to new tokenizer dir.")
    except Exception as e:
        logging.warning(f"change_vocabulary failed: {e}")

    # ========= Freeze Encoder (optional) ==========
    freeze_encoder = True
    if freeze_encoder:
        asr_model.encoder.freeze()
        asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen (BatchNorm / SE left trainable).")
    else:
        try:
            asr_model.encoder.unfreeze()
        except Exception:
            logging.info("Encoder unfreeze not available/necessary.")
        logging.info("Model encoder has been un-frozen")

    # --- Prepare a local copy of config and update tokenizer paths ---
    cfg = copy.deepcopy(asr_model.cfg)

    # Ensure tokenizer config updated
    with open_dict(cfg):
        cfg.tokenizer.dir = TOKENIZER_DIR
        cfg.tokenizer.type = "bpe"

        # Train dataset
        cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
        cfg.train_ds.batch_size = 8
        cfg.train_ds.num_workers = 8
        cfg.train_ds.pin_memory = True
        cfg.train_ds.use_start_end_token = True
        cfg.train_ds.trim_silence = True

        # Validation dataset
        cfg.validation_ds.manifest_filepath = VAL_MANIFEST
        cfg.validation_ds.batch_size = 8
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.use_start_end_token = True
        cfg.validation_ds.trim_silence = True

    # Apply the dataset configs to the model
    asr_model.setup_training_data(cfg.train_ds)
    asr_model.setup_validation_data(cfg.validation_ds)

    # optimizer/scheduler tweaks
    with open_dict(asr_model.cfg.optim):
        asr_model.cfg.optim.lr = 0.025
        asr_model.cfg.optim.weight_decay = 0.001
        # If warmup_steps exists by default, override or remove
        if "sched" in asr_model.cfg.optim and "warmup_steps" in asr_model.cfg.optim.sched:
            asr_model.cfg.optim.sched.warmup_steps = None
            asr_model.cfg.optim.sched.warmup_ratio = 0.10
            asr_model.cfg.optim.sched.min_lr = 1e-9

    # --- Trainer accelerator selection ---
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        logging.info("CUDA available — using GPU trainer.")
    else:
        accelerator = 'cpu'
        devices = 1
        logging.info("CUDA NOT available — using CPU trainer (this will be slow).")

    trainer = ptl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,  # you can change to desired frequency
    )

    # Attach trainer to the model
    asr_model.set_trainer(trainer)

    # Update the model's internal config to the modified cfg
    asr_model.cfg = cfg

    # Optional: create experiments directory
    exp_dir = os.path.join("experiments", f"lang-{LANGUAGE}")
    os.makedirs(exp_dir, exist_ok=True)
    logging.info(f"Experiment dir: {exp_dir}")

    # Fit
    try:
        trainer.fit(asr_model)
    except Exception as e:
        logging.error("Trainer.fit failed: %s", e)
        raise

    # Save final model
    save_path = f"Model-{LANGUAGE}.nemo"
    try:
        asr_model.save_to(save_path)
        logging.info(f"Model saved at path: {os.path.abspath(save_path)}")
    except Exception as e:
        logging.error("Failed to save model: %s", e)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Main failed: %s", e)
        raise