## Description

Fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) model for Belarusian language during 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
hosted by HuggingFace x Lambda.

The code in this repository is a modified version of code from 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event) repo.

## Fine-tuning todos:
* [ ] upload checkpoints & logs to HF
* [ ] perform evaluation of CV test set
* [ ] what type of lr scheduler is used?
* [x] stream training sample, download validation & test.<br>
      not downloading validation & test becuas not knowing there size and it's likely their size is large
* [x] Set `use_cache` to False since we're using gradient checkpointing, and the two are incompatible

## Notes:
* Whispers BasicTextNormalizer splits words containing apostrophe:
  ```python
  > from transformers.models.whisper.english_normalizer import BasicTextNormalizer
  > normalizer = BasicTextNormalizer()
  > normalizer("раз'яднаць")
  'раз яднаць'
  ```
* That's why `BelarusianTextNormalizer` (edited version of `BasicTextNormalizer`) was added to training script:
  ```python
  > from run_speech_recognition_seq2seq_streaming import BelarusianTextNormalizer
  > normalizer_be = BelarusianTextNormalizer()
  > normalizer_be("раз'яднаць")
  "раз'яднаць"
  ```
* Default Linear scheduler is used 
* Default Adam optimizer is used
* To save memory (and increase either model or batch_size) can experiment with:
    * using Adafactor instead of Adam.
      Adam requires two optimiser params per one model param, but Adafactor uses only one.
      > A word of caution: Adafactor is untested for fine-tuning Whisper, 
        so we are unsure sure how Adafactor performance compares to Adam!
    * using Adam 8bit from `bitsandbytes` module. 
      need to provide `optim="adamw_bnb_8bit"` param to `Seq2SeqTrainingArguments`