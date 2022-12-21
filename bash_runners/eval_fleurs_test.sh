python src/run_eval_whisper_streaming.py \
	--model_id="$1" \
	--language="be" \
	--dataset="google/fleurs" \
    --config="be_by" \
    --split="test" \
    --text_column="transcription" \
    --device="0" \
    --batch_size="32" \
    --streaming="True"
