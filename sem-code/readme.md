
# Terminal 1 — LLM

CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 ollama serve

# Terminal 2 — Judge

CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11436 OLLAMA_NUM_PARALLEL=4 ollama serve

deepseek-r1:14b

qwen3-coder:30b



setup:

ssh -t s4633466@silver.liacs.nl tmux new -As Ollama-deepseek-r1:14b

ssh -t s4633466@silver.liacs.nl tmux new -As Ollama-qwen3-coder:30b
