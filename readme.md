## Description

Fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) model for Belarusian language during 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
hosted by HuggingFace x Lambda.

The code in this repository is a modified version of code from 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event) repo.

## Fine-tuning todos:
* sync `run_debug.sh` with `run.sh`
* perform evaluation of fine-tuned model on CommonVoice test set

## Resuming training from exising checkpoint
When resuming training from existing checkpoint:
* learning rate gets reset if passing same paras to training script.<br>
  need to provide learning rate from the last step of previous run to continue
  training in a correct way.
* it's unclear whether decision on saving current model
  is made by comparing current metrics with metrics of the best checkpoint. I guess model with worse performance
  will not overwrite best model checkpoint already exising in the output dir, but need to double check.
* we can set `ignore_data_skip=True` Training argument not to 
  skip data items already passed to a model - that will save time on data loads.
    * it's unclear whether order of input items in the train set (that is shuffled) will be the same 
      across multiple reruns - i.e. it's unclear whether sampling is the same across reruns.
    * if the sampling is the same across reruns, `ignore_data_skip=True` will lead to same items been passed to a model
      in current run. it's OK if previous run ended with large step value on the last epoch.
      if not, the same elements from the same epoch will be passed to a model again.

## Questions:
* Why are there following lines in Data Collator?
  ```python
    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
        labels = labels[:, 1:]
    ```
* `tokenizer.bos_token_id` vs `model.config.decoder_start_token_id`.<br>
  which one to pass to Data Collator as `decoder_start_token_id` parameter?
* What checkpoint (best, I guess) is saved in the `output_dir`? 
  How is it overwritten when resuming training from existing checkpoint?

## Notes:
* using CommonVoice 11 dataset in a streaming way.<br>
  use `streaming=True` for train & validation & test.<br>
  as an alternative, we can use `streaming=False` for validation & test sets to save time on data processing.
  but the size of validation and test sets are unknown (need to check).
  it's likely they are going to be large - thus pre-download of these sets might not reduce 
  overall fine-tuning time compared to streaming mode.
* if using Google Colab, need to execute  `sudo chmod -R 777 .git` inside hf repo to 
  to set right permissions to be able to push trained models to HuggingFace Hub
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
* Need to set `use_cache` to False since we're using gradient checkpointing, and the two are incompatible
* Default Linear scheduler is used 
* Default Adam optimizer is used
* To save memory (and increase either model or batch_size) can experiment with:
    * using Adafactor instead of Adam.
      Adam requires two optimiser params per one model param, but Adafactor uses only one.
      > A word of caution: Adafactor is untested for fine-tuning Whisper, 
        so we are unsure sure how Adafactor performance compares to Adam!
    * using Adam 8bit from `bitsandbytes` module. 
      need to provide `optim="adamw_bnb_8bit"` param to `Seq2SeqTrainingArguments`