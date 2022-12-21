import argparse
import logging
import sys
import datetime
import os

import pandas as pd

from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, Audio
import evaluate

from belarusian_text_normalizer import BelarusianTextNormalizer


now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(filename=f'eval_{now_str}.log', mode='w')
    ],
)
logger.setLevel(logging.INFO)


wer_metric = evaluate.load("wer")
text_normalizer = BelarusianTextNormalizer()


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def normalise(sample, text_column: str):
    sample["reference_norm"] = text_normalizer(sample[text_column])
    return sample


def data(dataset,text_column: str):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference_norm": item["reference_norm"], 'reference': item[text_column]}


def clean_filename(filename: str):
    return filename.replace(os.path.sep, '_')


def main(args):
    logger.info(f'running evaluation script with following parameters: {args}')
    logger.info(f'using following text normalizer: {text_normalizer}')

    batch_size = args.batch_size
    whisper_asr = pipeline("automatic-speech-recognition", model=args.model_id, device=args.device)

    whisper_asr.model.config.forced_decoder_ids = (
        whisper_asr.tokenizer.get_decoder_prompt_ids(
            language=args.language, task="transcribe"
        )
    )

    logger.info('loading dataset')
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        use_auth_token=True,
    )

    # Only uncomment for debugging
    dataset = dataset.take(args.max_eval_samples)

    # TODO: probably no need in cast, because pipelien migh handle resampling internally. need to check
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise, fn_kwargs=dict(text_column=args.text_column))
    dataset = dataset.filter(is_target_text_in_range, input_columns=["reference_norm"])

    predictions = []
    predictions_norm = []
    references = []
    references_norm = []
    audio_paths = []

    logger.info('running inference')
    for out in whisper_asr(data(dataset, text_column=args.text_column), batch_size=batch_size):
        predictions.append(out["text"])
        predictions_norm.append(text_normalizer(out["text"]))
        references.append(out["reference"][0])
        references_norm.append(out["reference_norm"][0])
        audio_paths.append(out['path'][0])

    logger.info('computing metrics')
    wer = wer_metric.compute(references=references_norm, predictions=predictions_norm)
    wer = wer * 100

    logger.info('metrics computed')
    logger.info(f'WER: {wer}')

    if args.save_predictions is True:
        preds_fp = f'preds_{args.dataset}_{args.config}_{args.split}_{now_str}.tsv'
        preds_fp = clean_filename(preds_fp)
        logger.info(f'saving predictions to: "{preds_fp}"')
        preds_df = pd.DataFrame({
            'audio_path': audio_paths, 
            'prediction_norm': predictions_norm, 'reference_norm': references_norm,
            'prediction': predictions, 'reference': references, 
        })
        preds_df.to_csv(preds_fp, sep='\t', index=False)
    else:
        logger.info('save_predictions is False. will not save predictions to a file')

    if args.push_to_hub is True:
        logger.info(f'updating model card and pushing to HuggingFace Hub')
        evaluate.push_to_hub(
            model_id=args.model_id,

            metric_value=wer,
            metric_type="wer",
            metric_name="WER",

            dataset_name=args.dataset,
            dataset_type=args.dataset,
            dataset_config=args.config,
            dataset_split=args.split,
            
            task_type="automatic-speech-recognition",
            task_name="Automatic Speech Recognition"
        )
    else:
        logger.info('push_to_hub is False. will not update model card and push to HuggingFace Hub')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'` for the English split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'test'`",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Dataset column name containing target transcription of an audiofile"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Two letter language code for the transcription language, e.g. use 'en' for English.",
    )
    parser.add_argument(
        '--push_to_hub',
        type=bool,
        default=True,
        help="Whether to update model card and push changes to HuggingFace Hub"
    )
    parser.add_argument(
        '--save_predictions',
        type=bool,
        default=True,
        help="Whether to store predictions and target transcriptions to a file"
    )
    args = parser.parse_args()

    main(args)
