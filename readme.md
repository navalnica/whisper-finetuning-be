## Description

Fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) model for Belarusian language during 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
hosted by HuggingFace x Lambda.

The code in this repository is a modified version of code from 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event) repo.

## Fine-tuning todos:
* Learning rate:
  * max learning rate is not the same as LR passed as a parameter to training script. it is actually lower.
  * when resuming training, LR scheduling behaves incorrectly
* perform evaluation of fine-tuned model on CommonVoice test set
* check exact sizes of train, eval, test sets of CommonVoice 11

## Resuming training from exising checkpoint
When resuming training from existing checkpoint:
* it's better to save all `checkpoint-\d+` dirs. better not to rely on data saved to `output_dir` because:
  * not all data is saved to `output_dir`. e.g. following files are not saved to `output_dir`: 
    `optimizer.pt`, `rng_state.pth`, `scaler.pt`, `scheduler.pt`. so can't resume training in a correct way from
    data saved to `output_dir`
  * when resuming training from `output_dir` as a checkpoint dir, model saved to `output_dir` can be worse than
    previously save (need to investifate further. but such happened already)
* learning rate gets reset if passing same parameter value to training script as in previour run.<br>
  need to provide learning rate from the last step of previous run to continue
  training in a correct way
  * however even if passing learning rate from the last step, in the new run it has different value than expected
    * probably because last checkpont was chosen incorrectly
    * or learning rate is treated as a starting learning rate at step 0 and not on step X (where we resume).<br>
      need to try to pass same LR that was passes as a starting LR to the very first run
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
* What checkpoint (best, I guess) is saved in the `output_dir`? 
  How is it overwritten when resuming training from existing checkpoint?

### Prepended tokens
* Why are there following lines in Data Collator?
  ```python
    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
        labels = labels[:, 1:]
    ```
* `tokenizer.bos_token_id` vs `model.config.decoder_start_token_id`.<br>
  which one to pass to Data Collator as `decoder_start_token_id` parameter?
* Answer:
  * In this case, the two are equivalent. You can verify this:
    ```python
    print(tokenizer.bos_token_id)
    print(model.config.decoder_start_token_id)
    ```

  * Print Output:
    ```
    <|startoftranscript|>
    <|startoftranscript|>
    ```

  * Technically speaking, the decoder_start_token_id is the correct convention here. Before starting generating any tokens, we initialise the generate method with a starting token, which is the decoder_start_token_id. 
  See: https://huggingface.co/blog/how-to-generate. The decoder_start_token_id corresponds to the initial context word sequence, and is the zero'th token generated.

  * We remove this token from the encoded labels in the data collator because we always set the zero'th generated token to the decoder_start_token_id. If we leave the decoder_start_token_id as part of the label sequence, then we'll predict the decoder_start_token_id as the zero'th token, and again as the first token! Because we're always forcing it as the zero'th token, we don't need to predict it as the first token, and so we remove it from the target lables

  * These tokens are not forced in the generation process, and so we don't cut them in the data collator. We need to provide them to the model as target labels so that the model can learn the correct tasks from our data

  * The tokens correspond to the audio language, task (translate or transcribe) and whether to predict timestamps

  * We need to tell the model what language the audio corresponds to and what task it's performing during fine-tuning. This way, it learns what audio corresponds to what language, and the difference between transcribing audio vs translating it 

## Notes:
* using CommonVoice 11 dataset in a streaming way.<br>
  use `streaming=True` for train & validation & test.<br>
  as an alternative, we can use `streaming=False` for validation & test sets to save time on data processing.
  but the size of validation and test sets are unknown (need to check).
  it's likely they are going to be large - thus pre-download of these sets might not reduce 
  overall fine-tuning time compared to streaming mode.
* size of train set is ~370'000 audiofiles. if using `batch_size=64`, then
  1 epoch will have ~5782 steps. <br>
  Because of `--eval_steps="1000"` will use `--max_steps="6000"` instead of `--max_steps="5800"`
  to have evaluation metrics computed in the end of training.
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