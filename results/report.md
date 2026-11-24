| model | atco2_1h:test | uwb_atcc:test | atcosim:test |
|---|---|---|---|
| jlvdoorn/whisper-large-v3-atco2-asr-atcosim | 27.0%* | 55.2% | 19.5%* |
| jlvdoorn/whisper-large-v3-atco2-asr | 10.0%* | 34.6% | 12.5% |
| openai/whisper-large-v3-turbo | 46.6% | 74.1% | 19.9% |
| openai/whisper-large-v3 | 38.2% | 63.0% | 16.8% |
| openai/whisper-medium | 42.2% | 61.1% | 18.9% |
| openai/whisper-small | 55.1% | 83.8% | 29.8% |

WER after atc.normalize on both sides. * = model may have trained on this corpus.
