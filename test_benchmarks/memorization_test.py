from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import includes, model_graded_fact, match
from inspect_ai.solver import system_message, generate 


SYSTEM_MESSAGE = """
Respond with ONLY with the completed function, in full. Do NOT state your reasoning. Return the function as a string. do not include ```python before the function and ``` after the function.
""" 


def record_to_sample(record):
  return Sample(
      input=f"""Given the following code, complete the function: 
      
                {record['half_function']}
                
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
        dataset=dataset[:10],
        plan=[
          system_message(SYSTEM_MESSAGE),
          generate(),
        ],
        scorer=match()
    )
