from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import includes, model_graded_fact, model_graded_qa
from inspect_ai.solver import system_message, generate 


SYSTEM_MESSAGE = """
Respond with ONLY the answer to the question eg. "Accuracy". If more than one is used, list them eg. "Average F1, Accuracy". Do NOT state your reasoning.
"""


def record_to_sample(record):
  return Sample(
      input=record["metrics_only_prompts"],
      target=record["metrics_only_response"]
  )

@task
def pwc():
  
    dataset = hf_dataset(
        path="AlgorithmicResearchGroup/pwc_tasks_200",
        split="train",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True
    )
      
    return Task(
        dataset=dataset,
        plan=[
          system_message(SYSTEM_MESSAGE),
          generate(),
        ],
        scorer=model_graded_qa(
            model = ["openai/gpt-4"],
            partial_credit = True
            )
    )