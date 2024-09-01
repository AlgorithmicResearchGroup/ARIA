import argparse
import json
import pandas as pd
import random
from datasets import Dataset
from huggingface_hub import HfApi
from itertools import chain

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_jsonl(data, file_path):
    with open(file_path, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

def extract_model_info(json_data):
    extracted_info = []
    for dataset in json_data['datasets']:
        dataset_name = dataset['dataset']
        table_metric = dataset['sota']['metrics']
        for row in dataset['sota']['rows']:
            model_info = {
                'dataset': dataset_name,
                'model_name': row.get('model_name', ''),
                'paper_title': row.get('paper_title', ''),
                'metrics': row.get('metrics', {}),
                'table_metrics': table_metric
            }
            extracted_info.append(model_info)
    return extracted_info

def create_mixed_model_dataframe(df):
    mixed_models = []
    unique_areas = set(chain.from_iterable(filter(None, df['area'])))
    for area in unique_areas:
        same_area_models = df[df['area'].apply(lambda x: area in x)]['full_name'].tolist()
        different_area_models = df[df['area'].apply(lambda x: area not in x)]['full_name'].tolist()
        for model in same_area_models:
            other_same_area_models = [m for m in same_area_models if m != model]
            if len(other_same_area_models) >= 2:
                selected_same_area_models = random.sample(other_same_area_models, 2)
            else:
                continue
            if not different_area_models:
                continue
            selected_model_from_diff_area = random.choice(different_area_models)
            model_combo = [model] + selected_same_area_models + [selected_model_from_diff_area]
            random.shuffle(model_combo)
            areas = [area if area in df.loc[df['full_name'] == m, 'area'].values[0] else 'Outlier' for m in model_combo]
            mixed_models.append([model_combo, selected_model_from_diff_area, areas])
    mixed_model_df = pd.DataFrame(mixed_models, columns=['Model Combo', 'Outlier Model', 'Areas'])
    return mixed_model_df

def create_metrics_benchmark(data):
    df = pd.DataFrame(data)
    df['metrics_prompts'] = df.apply(lambda row: f"What metrics were used to measure the {row['model_name']} model in the {row['paper_title']} paper on the {row['dataset']} dataset?", axis=1)
    df['metrics_response'] = df['table_metrics'].apply(lambda x: ', '.join(x))
    return df[['metrics_prompts', 'metrics_response']]

def create_abstract_benchmark(data):
    df = pd.DataFrame(data)
    df['prompts'] = df['title'].apply(lambda x: f"Given the following ArXiv paper title {x}, write the abstract for the paper")
    return df[['prompts', 'abstract']]

def create_model_description_benchmark(data):
    df = pd.DataFrame(data)
    df['prompts'] = df['full_name'].apply(lambda x: f"Given the following machine learning model name: {x}, provide a description of the model")
    return df[['prompts', 'description']]

def create_research_area_benchmark(data):
    df = pd.DataFrame(data)
    df = df[~df['area'].apply(lambda x: 'General' in x or not x)]
    df['area'] = df['area'].apply(lambda x: list(set(x)))
    df['prompts'] = df['full_name'].apply(lambda x: f"Given the following machine learning model name: {x}, predict one or more research areas from the following list: [Computer Vision, Sequential, Reinforcement Learning, Natural Language Processing, Audio, Graphs]")
    
    # Convert area list to a string for easier comparison
    df['area_str'] = df['area'].apply(lambda x: ', '.join(sorted(x)))
    
    # Create a dictionary mapping area strings to labels
    area_to_label = {area: idx for idx, area in enumerate(df['area_str'].unique())}
    
    # Create label column
    df['label'] = df['area_str'].map(area_to_label)
    
    return df[['prompts', 'area', 'label']]

def create_odd_man_out_benchmark(data):
    df = pd.DataFrame(data)
    df['area'] = df['area'].apply(lambda x: [area['area'] for area in x['collections']])
    df = df[~df['area'].apply(lambda x: 'General' in x or not x)]
    df = df[~df['area'].apply(lambda x: 'Sequential' in x or not x)]
    df = df.dropna()
    try:
        mixed_model_df = create_mixed_model_dataframe(df)
    except Exception as e:
        print(f"Error in create_mixed_model_dataframe: {e}")
        return pd.DataFrame(columns=['prompts', 'models', 'label'])
    
    mixed_model_df['prompts'] = mixed_model_df['models'].apply(lambda x: f"Given the following list of machine learning models: {x}, select the one that least belongs in the list.")
    return mixed_model_df[['prompts', 'models', 'label']]

def create_similar_models_benchmark(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('dataset')['model_name'].apply(list).reset_index()
    filtered = grouped[grouped['model_name'].apply(len) > 1]
    filtered = filtered.dropna(subset=['dataset'])
    filtered['query_model'] = filtered['model_name'].apply(lambda x: x.pop(0) if x else None)
    filtered['prompts'] = filtered.apply(lambda row: f"Given the following machine learning model name: {row['query_model']}, and dataset: {row['dataset']}, provide a list of other models that have been benchmarked on that dataset", axis=1)
    return filtered[['prompts', 'model_name']]

def create_top_models_benchmark(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('dataset')['model_name'].apply(list).reset_index()
    filtered = grouped[grouped['model_name'].apply(len) > 1]
    filtered = filtered.dropna(subset=['dataset'])
    filtered['prompts'] = filtered['dataset'].apply(lambda x: f"Given the following benchmark dataset: {x}, provide a list of best performing models on that benchmark. Provide specific model names.")
    return filtered[['prompts', 'model_name']]

def create_dataset_description_benchmark(data):
    df = pd.DataFrame(data)
    df = df[~df['description'].str.contains("Click to add a brief description")]
    df['description'] = df['description'].str.split("\r\n\r\nSource").str[0]
    df['prompts'] = df['dataset_name'].apply(lambda x: f"Given the following benchmark dataset: {x}, provide a description of the benchmark dataset")
    return df[['prompts', 'description']]

def create_modality_benchmark(data):
    df = pd.DataFrame(data)
    df['modalities'] = df['modalities'].apply(eval)
    df = df[df['modalities'].apply(lambda x: len(x) <= 1)]
    desired_items = {'Graphs', 'Images', 'Texts', 'Tabular', 'Videos', 'Audio'}
    df = df[df['modalities'].apply(lambda x: bool(set(x) & desired_items))]
    df['prompts'] = df['name'].apply(lambda x: f"Given the following benchmark dataset: {x}, predict one or more research areas from the following list: ['Images', 'Graphs', 'Texts', 'Tabular', 'Videos', 'Audio']")
    
    # Convert modalities list to a string for easier comparison
    df['modalities_str'] = df['modalities'].apply(lambda x: ', '.join(sorted(x)))
    
    # Create a dictionary mapping modalities strings to labels
    modalities_to_label = {mod: idx for idx, mod in enumerate(df['modalities_str'].unique())}
    
    # Create label column
    df['label'] = df['modalities_str'].map(modalities_to_label)
    
    return df[['prompts', 'modalities', 'label']]

def save_benchmark(df, output_file, response_column):
    data = []
    for _, row in df.iterrows():
        output = {
            'input': [
                {"role": "system", "content": "You are a knowledgeable and helpful AI researcher"},
                {"role": "user", "content": row.get('prompts', 'No prompt available')}
            ],
            'ideal': row.get(response_column, 'No response available')
        }
        if 'label' in df.columns:
            output['label'] = int(row['label'])  # Ensure label is an integer
        data.append(output)
    save_jsonl(data, output_file)

def push_to_hub(df, dataset_name):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.filter(lambda x: all(v is not None for v in x.values()))
    dataset.push_to_hub(f"ArtifactAI/{dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Create ML benchmarks")
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output JSONL files")
    parser.add_argument("--push_to_hub", action="store_true", help="Push datasets to Hugging Face Hub")
    args = parser.parse_args()

    benchmarks = {
        "metrics": (f"{args.input_dir}/evaluation-tables.json", create_metrics_benchmark, extract_model_info),
        "abstract": (f"{args.input_dir}/papers-with-abstracts.json", create_abstract_benchmark, None),
        "model_description": (f"{args.input_dir}/methods.json", create_model_description_benchmark, None),
        "research_area": (f"{args.input_dir}/methods.json", create_research_area_benchmark, None),
        "odd_man_out": (f"{args.input_dir}/methods.json", create_odd_man_out_benchmark, None),
        "similar_models": (f"{args.input_dir}/evaluation-tables.json", create_similar_models_benchmark, extract_model_info),
        "top_models": (f"{args.input_dir}/evaluation-tables.json", create_top_models_benchmark, extract_model_info),
        "dataset_description": (f"{args.input_dir}/datasets.json", create_dataset_description_benchmark, None),
        "modality": (f"{args.input_dir}/datasets.json", create_modality_benchmark, None)
    }

    for name, (input_file, create_func, preprocess_func) in benchmarks.items():
        print(f"Creating {name} benchmark...")
        data = load_json(input_file)
        if preprocess_func:
            data = preprocess_func(data)
        df = create_func(data)
        output_file = f"{args.output_dir}/{name}_benchmark.jsonl"
        save_benchmark(df, output_file)
        print(f"Saved {name} benchmark to {output_file}")

        if args.push_to_hub:
            push_to_hub(df, f"{name}_benchmark")
            print(f"Pushed {name} benchmark to Hugging Face Hub")