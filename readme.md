## Description

### Whisper Fine-Tuning Event

* Fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) model for Belarusian language during 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
hosted by HuggingFace x Lambda.
* The code in this repository is a modified version of code from 
[Whisper fine-tuning Event](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event) repo.

### Models and demo
You can find fine-tuned models and their demos here:
|Model type|CommonVoice11 test WER|CommonVoice11 dev WER|Model link|Model demo|
|---|---|---|---|---|
|Whisper Small|6.79%|6.367%|[ales/whisper-small-belarusian](https://huggingface.co/ales/whisper-small-belarusian)|[ales/whisper-small-belarusian-demo](https://huggingface.co/spaces/ales/whisper-small-belarusian-demo)|
|Whisper Base|-|6.79%|[ales/whisper-base-belarusian](https://huggingface.co/ales/whisper-base-belarusian)|-|


## Tips:
* start with a port worwarding to monitor Tensorboard logs on local computer:
  ```
  ssh <remote-address> -L <local_port>:localhost:<remote_tensorboard_port>
  ```
* Train with redirecting output to a file using `tee`:
  ```
  source src/run.sh 2>&1 | tee train_run_<run_number>.log
  ```

## TODO
* upload CV11 eda
* read more about chunking and striding during inference:
  * https://discord.com/channels/879548962464493619/1050962020004020246/1054518630999666709
  * https://github.com/huggingface/transformers/pull/20104
* check exact sizes of train, eval, test sets of CommonVoice 11
* fill TODOs in Notes section with answers and discussions from a Discord
  * https://discord.com/channels/879548962464493619/1045270618729361439/1050429189947408475
* train & evaluate on extended Common Voice 11 dataset (with multiple voicings allowed for same sentence)
* view [event intro video](https://www.youtube.com/watch?v=1cVBLOMlv3w)

### Fine-tuning
* We fine-tune Whisper that uses constant 30-seconds windows on Common Voice data that is mostly 5-8 seconds long.<br>
  Thus, most of input features are padded values (zeros).<br>
  Fine-tuned model produces lots of repetitions on long audiofile (>8s long).<br>
  To make model be able to predict for any audiofile with length <30 seconds, 
  we need to combine multiple short audiofiles and transcriptions into 30-seconds length audiofile.<br>
  However, as combined transcriptions would not be related to each other, this may decrease model performance.
* upd all .sh scripts to use `tee` and create `logs` dir
* logs are printed only right before the evalutaion:<br>
  ```
  --logging_steps="50"
  --eval_steps="1000"
  ```

### Evaluation & Inference
* Can we use chunking and striding during inference with transformer encoder-decoder models?<br>
  we definitely can with CTC models. Check [this guide](https://huggingface.co/blog/asr-chunking)<br>
  In CTC case we predict probabilities of each character for each time point.<br>
  In contrast, for transformer encoder-decoder models we predict entire sequence without direct
  correspondence to time stamps.
* Check whether hallucinations happen during inference without chunking (test long audiofiles)
* Check if model hallucinates on any data from Common Voice dataset (validation & test & probably samples from train).
  It was fine-tuned on Common Voice.

#### Fleurs
* Fleurs dataset contains multiple voicings of same sentences - confirm that. e.g.:
  * `audio_path='3208476310630955776.wav'`
  * `audio_path='18432407656171360869.wav'`
  * `audio_path='4977735674010378775.wav'`
* Can't use Whisper's text normalization, because text inside parenthesis was voiced in Fleurs.
  see `audio_path = '10408937009990230772.wav'`
* Transcriptions in Fleurs contain:
  * unnormalized text with numbers written as digits: e.g. `audio_path='15454108795071358409.wav'`
  * non-belarusian words. e.g. `audio_path='3083565517142153888.wav'`
  * cyrillic transcription of English words. e.g. `audio_path='8551756870735878885.wav'`
  * abbreviations. e.g. `audio_path='13661479120400313348.wav'`.<br>
    `reference_norm="бесправадная сетка стандарту 802 11n выкарыстоўвае частоты 2 4 ггц і 5 0 ггц "`
* sometimes transcriptions do not match voiced text. e.g.:
  * `audio_path="17735478096436983770.wav"`
  * `reference_norm="у 2005 фінансавым годзе..."`
  * voiced text = "у 2005 годзе фінансавым годзе..."
* There is a 
  [Discussion on Fleurs issues for Armenian](https://huggingface.co/datasets/google/fleurs/discussions/8). 
  Can create similar Discussion for Belarusian and propose to remove problematic examples.

### Demo
* Using `chunk_length_s` is very experimental with seq2seq models. The results will not necessarily be entirely accurate and will have caveats. More information: https://github.com/huggingface/transformers/pull/20104. Ignore this warning with pipeline(..., ignore_warning=True)
* /home/user/.pyenv/versions/3.8.9/lib/python3.8/site-packages/transformers/generation/utils.py:1134: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation)
* /home/user/.pyenv/versions/3.8.9/lib/python3.8/site-packages/transformers/pipelines/base.py:1043: UserWarning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset


## Resuming training from exising checkpoint
When resuming training from existing checkpoint:
* TODO: probably need to load `optimizer.pt` and `scaler.pt` from checkpoint before resuming training.
  otherwise, I guess, we
  * reinitialize optimizer and loose history of parameters momentum (exponential weighted average)
  * scale loss incorrectly
* when using streaming, epoch will get reset to 0. that means order of items passed to a model would be the same,
  if the seed does not change. actual train_dataloader seed would be: 
  `train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)`
* it's better to save all `checkpoint-\d+` dirs. better not to rely on data saved to `output_dir` because:
  * not all data is saved to `output_dir`. e.g. following files are not saved to `output_dir`: 
    `optimizer.pt`, `rng_state.pth`, `scaler.pt`, `scheduler.pt`. so can't resume training in a correct way from
    data saved to `output_dir`
  * when resuming training from `output_dir` as a checkpoint dir, model saved to `output_dir` can be worse than
    previously save (need to investifate further. but such happened already)
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

### Scheduling Learning Rate when resuming training
* When resuming training, total number of optimization steps changes
* Usinng default LR-scheduler (linear with warmup), will result in unexpected LR changes
* To explicitly control the maximum LR (after warmup is finished) and the LR in the end of training
  I've subclassed `transformers.Trainer` class and overriden `create_scheduler()` function in 
  `custom_trainer.Seq2SeqTrainerCustomLinearScheduler`
* EDA on controlling LR scheduling could be found in `eda/trainer_lr_scheduler.ipynb` notebook


## Questions:
* What checkpoint (best, I guess) is saved in the `output_dir`? 
  How is it overwritten when resuming training from existing checkpoint?
* why dataset loading crashes when using `num_proc > 0`?
* does `ShuffleCallback` work with StreamingDataset? it reshuffles data `on_epoch_begin()`,
  but does StreamingDataset have any epochs?
* does streaming mode support parallel data load and processing?<br>
  when using non-streaming mode we can use `dataset.map(..., num_proc=<num_proc>)`
* I got CUDA out of memory error when tried to launch a second training run for Whisper Small model.
  training params are almost the same: `--per_device_train_batch_size="64"`
  the only thing changed is that now evaluation dataset now doesn't use streaming.


## Notes:
* Common Voice 11 dataset 
  [uploaded to HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
  has only single voicing of each sentence in each split (train, validation, test).<br>
  Much more audiofiles should be available on Common Voice so that each sentence is voiced multiple times by different people
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
* Log tracking in Jupyter (not working) and in bash (works as expected with `tee`)
* Loggers in `run_speech.....py` do not control `transformers` and `datasets` loggers. 
  can't redirect their outputs using handlers. it's better and easier to redirect output in a bash
* to evaluate on `google/fleurs` dataset had to downgrade `numba` from `0.56.4` to `0.56.3`, then install `librosa`
  (strange, because `librosa` should have been installed when `pip install -r ~/whisper-finetuning-be/requirements.txt`
  was run) and then upgrade back to `numba==0.56.4` because couldn't `import numba` when it was `0.56.3`
* Need to set `use_cache` to False since we're using gradient checkpointing, and the two are incompatible
* Default Linear scheduler is used 
* Default Adam optimizer is used

### Hallcunations, inference, chunking and striding
* observed when running inference for long audiofiles 
  using transformers `autumatic-speech-recognition` pipeline with `chunk_length_s=30`
* Probable hallucinations reasons:
  * running model on audiochunks that are longer compared to ones the model was trained on
  * overfit on Common Voice 11 data. poor generalization to real-world audio
  * audiofiles contain large segments of silence
* TODO: check whether hallucinations happen during inference without chunking
* decreasing chunk size from 30 to `chunk_length_s=8` helped. Model doesn't hallucinate.
* when using `chunk_length_s=5` hallucinations appear again
* model struggles to predict correctly close to chunk borders.<br>
  near chunk borders model often predicts periods (`.`) instead of letters or may even skip some words.<br>
  striding does not seem to help very much.
* however, `chunk_length_s=8` and `stride_length_s=1` seem to produce best transcriptions so far.
* default striding value `stride_length_s = chunk_length_s / 6` also performs well, 
  but a bit worse than `stride_length_s=1` (at least for one particular test sample).
* using large striding (2, 3, 4) makes predictions worse
* I wonder whether we even can use chunking for transformer-based encoder-decoder architecture like in Whisper

### Other hallucination notes
* from https://discord.com/channels/879548962464493619/1052230104304074793/1052282095596224632:
  * In my experience the model hallucinates in one of two scenarios:
    * When there is long period of silences in an audio
    * When the model was trained with a data mismatch i.e. there were cases where the audio file and the transcription does not match.
  * Just a headsup, the Whisper model can only handle 30sec of audio in one go. 
    During training any audio longerthan 30sec is automatically truncated.
    While we truncate the audio, the text transcript still remains the same. 
    This results in a mismatch between the audio and the reference transcription.
  * I'd recommend filtering out all the data points which are longer than 30 sec from your train set.
* from https://discord.com/channels/879548962464493619/1052230104304074793/1052797714678693950:
  * Turns out the problem was caused by transformers version(Working fine: 4.25.1 and higher(master branch as well)
  * Casing hallucinations: 4.24.0
  * I think the problem was caused by tokenizer, though I havent investigated further (any way the behaviour is fixed)
  * After switching to 4.25.1 medium training went much better

### Logs not printed when expected
* Train logs are printed only before start of a validation. 
  During training they are not printed to a stdout.
  All worked fine in a Colab.
* No progressbar for validation (at least when using streaming and iterable dataset). 
  possible reason is that when using streaming, the dataset len in unknown.
* Evaluation metrics get printed to stdout only before the next validation call.
  All worked fine in a Colab.
* Possible reason: usage of `... | tee file.log`. But it's unlikely

### Text normalization
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

### Different batch sizes for train and evaluation:
* Theoretically you can use a larger batch size for evaluation vs training! 
* Training: we do a forward pass, storing all the activations, and then a backwards pass, storing all the gradients
* Inference (evaluation): we only do a forward pass, and don't store any activations
* So the memory required for evaluation is much lower than it is for training 
  (we're only doing the forward pass and not storing any values) 
* In my experience, altering the eval batch size has little effect on eval speed -> 
  I set it to a lower value as this tends to give a more responsive progress bar 
  when evaluating in non-streaming mode (the bar updates faster and more frequently) 

### Slow inference. Long evalutaion compared to training:
* Slower inference is an inherent limitation of the sequence-to-sequence architecture. 
  The auto-regressive decoding means that you have to do as many decoder forward passes as tokens generated. 
* This is much slower than CTC, where you do a single encoder forward pass
* Note that 1 evaluation step **will take much longer** than 1 training step, even with the same batch sizes.
  * With training, we do one forward pass of the encoder, one forward pass of the decoder, 
    one backward pass of the decoder and one backward pass of the encoder (=4 passes total):<br>
    ```
    audio -> encoder -> decoder -> labels
              encoder <- decoder <- loss 
    ```
  * During evaluation we do one forward pass of the encoder, and then auto-regressively generate tokens in the decoder. 
    Here, we do as many forward passes of the decoder as tokens generated. 
    So in total, we do one forward pass of the encoder, and N forward passes of the decoder, 
    where N is the number of tokens generated (can be up to the max length, which is 448...). 
    You can see that for 4 or more generated tokens, evaluation is going to be slower than training:<br>
    ```
    audio -> encoder -> decoder -> decoder -> decoder -> ... -> decoder -> end of sentence token
    ```
* I've made a bit of a simplification here in saying that one forward pass 
  takes the same amount of time as one backward pass, but for the purpose of illustrating,
  this demonstrates the point why evaluation is much slower than training 
* Essentially it doesn't really matter what you set your eval batch size as we're not aggregating any statistics 
  over the eval batch (in contrast during training we evaluate a true gradient value based on a given batch). 
  * Since we just do a forward pass, we could even run eval with a batch size of 1 and get exactly the same results!
  * Because we don't get much of an improvement with batch sizes beyond around 8, it's set somewhat arbitrarily

### Ways to decrease evaluation time during fine-tuning:
* reduce `generation_max_length` param:
  * During training, we can limit the generation max length to a lower number to cut-off the generation 
    after fewer tokens (e.g. 40). This will give worse results during training, 
    but we can still infer the evolution of WER performance over training. 
  * For the final eval step, we can bump up the generation max length back up to 448. 
  * WER performance varies monotonically with generation max length 
    (WER can only stay equal or improve by increasing generation max length), 
    so we know that our final eval WER will be less than (improved) or equal to the WER during training
* We can evaluate at less frequent eval_steps: this reduces the number of times we have to perform evaluation

### Decrease inference time more generally
* PyTorch 2.0 and compiling the model could get you a decent speed-up 
  (https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/#hugging-face-models)
* Downcasting to fp16

### Memory saving and training larger models:
To save memory (and increase either model or batch_size) can experiment with:
* using Adafactor instead of Adam.
  Adam requires two optimiser params per one model param, but Adafactor uses only one.
  > A word of caution: Adafactor is untested for fine-tuning Whisper, 
    so we are unsure sure how Adafactor performance compares to Adam!
* using Adam 8bit from `bitsandbytes` module. 
  need to provide `optim="adamw_bnb_8bit"` param to `Seq2SeqTrainingArguments`
* use `deepspeed`. scripts are there in 
  [Whisper fine-tuning Event repo](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
* load the model and processor in 8bit mode:
  ```python
  from transformers import WhisperForConditionalGeneration, WhisperProcessor
  model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large", device_map="auto", load_in_8bit=True)
  processor = WhisperProcessor.from_pretrained("openai/whisper-large", load_in_8bit=True)
  ```
  inference loop:
  ```python
  for data in dataset:
    inputs = processor.feature_extractor(data["audio"]["array"], return_tensors="pt", sampling_rate=16_000).input_features.half().to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)[0]
    print(text)
  ```
  * 8bit will slower iference compared to full/half-precision
  * But the memory saving you get is immense (up to 4x vs full-precision).<br>
    This is the recommended approach when you're limited on VRAM.<br>
    If you care about inference speed, still to full precision

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

  