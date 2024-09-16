from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import includes, model_graded_fact, model_graded_qa
from inspect_ai.solver import system_message, generate 



DEFAULT_MODEL_GRADED_QA_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""

def record_to_sample(record):
  return Sample(
      input=record["prompt"],
      target=record["description"]
  )

@task
def pwc():
  
    dataset = hf_dataset(
        path="AlgorithmicResearchGroup/predict_model_description_given_model_name",
        split="train",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True
    )
      
    return Task(
        dataset=dataset[:10],
        plan=[
          generate(),
        ],
        scorer=model_graded_qa(
            model = ["openai/gpt-4"]
        )
    )