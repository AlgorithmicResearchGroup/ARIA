# ARIA Benchmarks (AI Research Intelligence Assessment)

ARIA Benchmarks is a suite of closed-book benchmarks designed to assess a models knowledge and understanding of machine learning research and methodologies

## Overview

The ARIA Benchmarks evaluate various aspects of AI research knowledge, including:

- Dataset Modality QA
- Model Modality QA
- Odd Model Out Benchmark
- PWC Metrics 1000
- PWC Metrics:Result 1000


These benchmarks are derived from the Papers With Code dataset, ensuring they reflect current trends and developments in AI research.

## Benchmark Types


### 1. Dataset Modality QA
Tests the model's ability to identify the modalities of different datasets.

### 2. Model Modality QA
Tests the model's ability to identify the modalities of different models.

### 3. Odd Model Out Benchmark
Tests the model's ability to identify the odd model out from a list of models.

### 4. PWC Metrics 1000
Tests the model's ability to identify metric, given a model and dataset.

### 5. PWC Metrics:Result 1000
Tests the model's ability to identify the metric result, given a model and dataset.


| Metrics                     | GPT-4o | GPT-4 | GPT-3.5-Turbo | Claude-Opus | Claude-Sonnet | Claude-Haiku | Gemini-Pro |
|-----------------------------|--------|-------|---------------|-------------|---------------|--------------|------------|
| Dataset Modality QA         | 0.685  | 0.62  | 0.477         | 0.719       | 0.699         | 0.716        | 0.458      |
| Model Modality QA           | 0.853  | 0.82  | 0.731         | 0.798       | 0.748         | 0.788        | 0.756      |
| Odd Model Out Benchmark     | 0.562  | 0.456 | 0.354         | 0.451       | 0.369         | 0.307        | 0.371      |
| PWC Metrics 1000            | 0.53   | 0.466 | 0.392         | 0.497       | 0.422         | 0.273        | 0.373      |
| PWC Metrics:Result 1000     | 0.025  | 0.03  | 0.085         | 0.065       | 0.02          | 0.025        | 0.055      |


## Benchmark Format

Each benchmark in the ARIA suite is provided in JSONL format, with each entry containing:

- `input`: A prompt or question about AI research
- `ideal`: The expected response or answer

## Usage

These benchmarks can be used to:

1. Evaluate the AI research knowledge of large language models
2. Assess the effectiveness of AI education and training programs
3. Identify gaps in an AI system's understanding of machine learning concepts
4. Compare different AI models' grasp of current research trends

## Data Source

The ARIA Benchmarks are created using data from Papers With Code, a free and open resource for machine learning papers, code, and evaluation tables.

## Contributing

We welcome contributions to improve and expand the ARIA Benchmarks. Please submit pull requests or open issues for suggestions and improvements.

## License

The ARIA Benchmarks are released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use the ARIA Benchmarks in your research, please cite:

@misc{aria-benchmarks,
    title={ARIA Benchmarks: AI Research Intelligence Assessment},
    year={2023},
    url={https://github.com/yourusername/ARIA-Benchmarks}
}

## Contact

For questions or feedback about the ARIA Benchmarks, please open an issue in this repository.