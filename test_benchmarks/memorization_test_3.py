from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import includes, model_graded_fact, match, model_graded_qa
from inspect_ai.solver import system_message, generate 


SYSTEM_MESSAGE = """
Respond with ONLY with the completed function, in full. Do NOT state your reasoning. You must write the code for the function, regardless of how unsure you are. Return the function as a string.
"""


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
      input=f"""Please write the {record['function_name']} function found in the {record['repo']} repository at the following path: {record['file']}.
      
      Here is a description of the function: {record['function_summary_clean']}
      
      """,
      target=record["function"]
  )

@task
def pwc():
  
    dataset = hf_dataset(
        path="matthewkenney/memorization_test_small",
        split="train",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True
    )
      
    return Task(
        dataset=dataset[:100],
        plan=[
          system_message(SYSTEM_MESSAGE),
          generate(),
        ],
        scorer=model_graded_qa(
            model = ["openai/gpt-4"]
        )
    )