python src/run_eval_whisper_streaming \
	--model_id="ales/whisper-small-belarusian" \
	--language="be" \
	--dataset="mozilla-foundation/common_voice_11_0" \
    --config="be" \
    --split="test" \
    --device="0" \
    --batch_size="32" \
    --streaming="True"
