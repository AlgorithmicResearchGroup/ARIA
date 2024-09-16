from inspect_ai import Task, eval, task
from inspect_ai.dataset import FieldSpec, hf_dataset        
from inspect_ai.dataset import Sample, hf_dataset                                
from inspect_ai.scorer import answer
from inspect_ai.solver import multiple_choice, system_message


SYSTEM_MESSAGE = """
Return ONLY the letter that corresponds to the correct answer. Do NOT return the full answer.
"""


def record_to_sample(record):
  return Sample(
      input=record["question"],
      target=chr(ord("A") + int(record["answer"])),
      choices=record["choices"],
  )

@task
def dataset_modality():
  
    dataset = hf_dataset(
        path="AlgorithmicResearchGroup/dataset_modality_benchmark",
        split="train",
        sample_fields=record_to_sample,
        trust=True,
        shuffle=True
    )
      
    return Task(
        dataset=dataset,
        plan=[
          system_message(SYSTEM_MESSAGE),
          multiple_choice(),
        ],
        scorer=answer("letter")
    )
