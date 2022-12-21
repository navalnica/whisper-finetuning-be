python src/run_eval_whisper_streaming.py \
	--model_id="$1" \
	--language="be" \
	--dataset="mozilla-foundation/common_voice_11_0" \
    --config="be" \
    --split="test" \
    --text_column="sentence" \
    --device="0" \
    --batch_size="32" \
    --streaming="True"