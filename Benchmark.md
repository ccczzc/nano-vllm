## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4090 (48GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine     | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------------------------------|---------|----------|-----------|
| Nano-vLLM with `flash_attn` library      | 133,966 | 19.86  | 6745.49   |
| Nano-vLLM with `triton` attention kernel | 133,966 | 22.08  | 6067.09   |