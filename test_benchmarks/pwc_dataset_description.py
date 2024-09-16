from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import includes, model_graded_fact, model_graded_qa
from inspect_ai.solver import system_message, generate 



def record_to_sample(record):
  return Sample(
      input=record["prompt"],
      target=record["description"]
  )

@task
def pwc():
  
    dataset = hf_dataset(
        path="AlgorithmicResearchGroup/predict_dataset_description_given_dataset_name",
        split="train",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True
    )
      
    return Task(
        dataset=dataset,
        plan=[
          generate(),
        ],
        scorer=model_graded_qa(
            model = ["anthropic/claude-3-opus-20240229"]
        )
    )