mkdir -p logs

python src/run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="ales/whisper-small-belarusian" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="be" \
	--language="be" \
	--train_split_name="train" \
	--eval_split_name="validation" \
	--model_index_name="Whisper Small Belarusian" \
    \
	--max_steps="6000" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="50" \
	--logging_first_step \
	--learning_rate="3e-5" \
	--learning_rate_end="1e-5" \
	--warmup_steps="0" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--gradient_checkpointing \
	--fp16 \
    \
	--shuffle_buffer_size="500" \
	--generation_max_length="225" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
    \
	--do_train \
	--do_eval \
	--ignore_data_skip \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming_train="True" \
	--streaming_eval="False" \
	--seed="43" \
	--use_auth_token \
	--push_to_hub="False" 2>&1 | tee "logs/train_$(date +"%Y%m%d-%H%M%S").log"
