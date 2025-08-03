@echo off
set OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE

python run.py sim ^
    --market-data-path data/05_model_input_log/env_data.pkl ^
    --start-time 2023-01-01 ^
    --end-time 2023-12-31 ^
    --run-model train ^
    --config-path config/fpt_ollama_config.toml ^
    --checkpoint-path data/06_train_checkpoint ^
    --result-path data/09_results/fpt_ollama_results
