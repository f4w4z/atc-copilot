"""Fine-tune Whisper on public ATC corpora.

    python -m atc.finetune --model openai/whisper-small --out runs/small-v1
    python -m atc.finetune --model openai/whisper-tiny --out runs/smoke --max-steps 20 \
        --max-train-samples 64 --eval-samples 16 --eval-every 10 --batch-size 4

Targets are the corpus references, which are lowercase spoken forms, so the model learns
to emit "flight level one hundred" rather than "FL100". Validation WER is reported per eval set
(eval_<set>_wer) with the same normaliser as atc.eval_wer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jiwer
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from atc import data
from atc.normalize import normalize

MAX_LABEL_TOKENS = 448
MAX_AUDIO_SECONDS = 30.0


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = [{"input_ids": f["labels"]} for f in features]
        lab = self.processor.tokenizer.pad(labels, return_tensors="pt")
        ids = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        # The model prepends the decoder start token itself.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


class Featurize:
    """Log-mel features computed on the fly. Caching them with .map() would cost
    ~1 MB per utterance (80 x 3000 float32), about 18 GB for the default training set."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, batch):
        arrays = [a["array"] for a in batch["audio"]]
        feats = self.fe(arrays, sampling_rate=data.SAMPLING_RATE).input_features
        return {"input_features": list(feats), "labels": batch["labels"]}


class Tokenize:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        # References are already spoken form; train on them as written and normalise only when scoring.
        return {"labels": [self.tok(" ".join(t.split())).input_ids for t in batch["text"]]}


def prepare(ds, processor, num_proc: int | None):
    """Tokenise labels and drop over-long utterances; audio stays undecoded until batching."""
    keep = {"audio", "duration"}
    ds = ds.map(Tokenize(processor.tokenizer), batched=True, num_proc=num_proc,
                remove_columns=[c for c in ds.column_names if c not in keep])
    ds = ds.filter(
        lambda d, l: d <= MAX_AUDIO_SECONDS and len(l) <= MAX_LABEL_TOKENS,
        input_columns=["duration", "labels"],
    )
    return ds.remove_columns(["duration"])


class ATCTrainer(Seq2SeqTrainer):
    """Evaluation runs in-process. On Windows every dataloader worker is a spawned
    process that reserves ~2.4 GB of commit memory, and the stock Trainer started a
    fresh set per eval round without releasing the old ones, which exhausted the
    commit limit after ~1,500 steps."""

    def __init__(self, *args, eval_workers: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_workers = eval_workers

    def get_eval_dataloader(self, eval_dataset=None):
        ds = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(ds, str):
            ds = self.eval_dataset[ds]
        return DataLoader(ds, batch_size=self.args.per_device_eval_batch_size,
                          collate_fn=self.data_collator, num_workers=self.eval_workers,
                          pin_memory=torch.cuda.is_available())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--out", required=True)
    p.add_argument("--train", nargs="+", default=["uwb_atcc:train", "atcosim:train"])
    p.add_argument("--eval", nargs="+", default=["uwb_atcc:test", "atco2_1h:test"])
    p.add_argument("--max-train-samples", type=int, default=None, help="per training set")
    p.add_argument("--eval-samples", type=int, default=400, help="fixed subset per eval set")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--num-proc", type=int, default=None, help="preprocessing workers")
    p.add_argument("--report-to", default="none", help="e.g. tensorboard, wandb")
    p.add_argument("--workers", type=int, default=2, help="training dataloader workers (audio decoding)")
    p.add_argument("--eval-workers", type=int, default=0, help="eval dataloader workers")
    p.add_argument("--save-only-model", action="store_true",
                   help="checkpoints without optimizer state: ~1/3 the disk, but no exact resume")
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out")
    args = p.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model)
    multilingual = not args.model.endswith(".en")
    if multilingual:
        processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    if multilingual:
        model.generation_config.language = "english"
        model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if args.freeze_encoder:
        model.freeze_encoder()

    train = concatenate_datasets([
        prepare(data.load(s, max_samples=args.max_train_samples, seed=1), processor, args.num_proc)
        for s in args.train
    ]).shuffle(seed=0).with_transform(Featurize(processor.feature_extractor))
    evals = {
        data.parse_spec(s)[0]: prepare(data.load(s, max_samples=args.eval_samples), processor, args.num_proc)
        .with_transform(Featurize(processor.feature_extractor))
        for s in args.eval
    }
    print(f"train {len(train)} utterances; eval " + ", ".join(f"{k}={len(v)}" for k, v in evals.items()))

    tok = processor.tokenizer

    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tok.pad_token_id
        hyps = [normalize(t) or "<empty>" for t in tok.batch_decode(pred.predictions, skip_special_tokens=True)]
        refs = [normalize(t) for t in tok.batch_decode(label_ids, skip_special_tokens=True)]
        keep = [i for i, r in enumerate(refs) if r]
        return {"wer": jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep])}

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_grad_ckpt,
        bf16=bf16,
        fp16=cuda and not bf16,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_steps=args.eval_every,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{next(iter(evals))}_wer",
        greater_is_better=False,
        report_to=args.report_to,
        dataloader_num_workers=args.workers,
        save_only_model=args.save_only_model,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = ATCTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        eval_workers=args.eval_workers,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
