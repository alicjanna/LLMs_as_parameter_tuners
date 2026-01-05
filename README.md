# LLMs as parameter tuners
This is the code repository associated with the paper:
"Tuning metaheuristic parameters with the use of Large Language Models"

![Method overview](teaser.png)

## Required Libraries
pandas, numpy, mealpy, mock, time, anthropic, openai, google.generativeai, mistralai

## Data
Folders with problem instances:
* instances_02_TSP
* instances_03_JSSP
* instances_04_GCP

## How to use it?
- parameters_main.xlsx - should intitially have runs as rows and parameter values as columns
- talk_with_llm.py goes thorugh the file and collects parameter suggestions from various llms
- then run TSP.py or JSSP.py or GCP.py accordingly to the needs (RUN_TYPE = 'Initial')
- calculate metrics and update prompts for feedback run
- talk_with_llm.py for Feedback
- run TSP.py or JSSP.py or GCP.py
- calculate final statistics

## Citation
```bibtex
@article{martinek2026tuning,
  title={Tuning metaheuristic parameters with the use of Large Language Models},
  author={Martinek A., Bartuzi-Trokielewicz E., Łukasik S., Gandomi A.},
  journal={},
  year={2026}
}
