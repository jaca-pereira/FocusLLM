import os
import csv
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import json

def process_json_file_and_calculate_pvalue(json_file_path, baseline_json_path):
    correct_before = 0
    correct_after = 0
    incorrect_before = 0
    incorrect_after = 0

    # Open and read the baseline JSON file line by line
    with open(baseline_json_path, 'r') as baseline_file:
        baseline_data = {item['video_id']: item for item in map(json.loads, baseline_file)}

    # Open and read the ablation JSON file line by line
    with open(json_file_path, 'r') as ablation_file:
        for line in ablation_file:
            ablation_data = json.loads(line)
            video_id = ablation_data['video_id']

            # Retrieve corresponding baseline data
            baseline_entry = baseline_data.get(video_id)

            if baseline_entry:
                for question, baseline_question in zip(ablation_data['questions'], baseline_entry['questions']):
                    answer = question['answer']
                    response_before = baseline_question['response']
                    response_after = question['response']

                    if answer == response_before and answer == response_after:
                        correct_before += 1
                    elif answer == response_before and answer != response_after:
                        incorrect_after += 1
                    elif answer != response_before and answer == response_after:
                        correct_after += 1
                    else:
                        incorrect_before += 1

    # Construct the contingency table
    table = np.array([[correct_before, incorrect_after],
                      [correct_after, incorrect_before]])

    # Apply McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    # Extract the p-value
    p_value = result.pvalue

    # Determine whether to reject the null hypothesis
    reject_null = p_value < 0.05

    # Return the p-value and the reject_null flag
    return p_value, reject_null

def save_ablation_results_to_csv(data, filename='ablation_results.csv'):
    # Define the header
    header = ['nr_frames', 'layer(s)', 'smooth', 'p_value', 'reject_null', 'p_value_sub', 'reject_null_sub']

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        # Write the data rows
        for row in data:
            writer.writerow(row)

    print(f"Results saved to {filename}")

def process_directory_structure(root_dir, model_name, baseline_dir_name='videomme_replicate'):
    results = []

    # Loop through each benchmark directory
    for benchmark_dir in os.listdir(root_dir):
        benchmark_path = os.path.join(root_dir, benchmark_dir)
        if os.path.isdir(benchmark_path):
            # Parse the directory name
            parts = benchmark_dir.split('_')
            smooth = parts[-2] == 'True'
            nr_frames = int(parts[-1])
            layers_segments = parts[1:-2]
            layers = [int(layer) for layer in layers_segments[:len(layers_segments) // 2]]

            # Directory where the ablation answer JSON files are stored
            answers_dir = os.path.join(benchmark_path, 'answers', model_name)

            # Paths to the ablation JSON files
            merge_json_path = os.path.join(answers_dir, 'merge.json')
            merge_sub_json_path = os.path.join(answers_dir, 'merge_sub.json')

            # Paths to the baseline JSON files
            baseline_answers_dir = os.path.join(root_dir, baseline_dir_name, 'answers', model_name)
            baseline_merge_json_path = os.path.join(baseline_answers_dir, 'merge.json')
            baseline_merge_sub_json_path = os.path.join(baseline_answers_dir, 'merge_sub.json')

            if os.path.exists(merge_json_path) and os.path.exists(baseline_merge_json_path):
                # Process merge.json
                p_value, reject_null = process_json_file_and_calculate_pvalue(merge_json_path, baseline_merge_json_path)
            else:
                print(f"Warning: {merge_json_path} or {baseline_merge_json_path} does not exist.")
                continue

            if os.path.exists(merge_sub_json_path) and os.path.exists(baseline_merge_sub_json_path):
                # Process merge_sub.json
                p_value_sub, reject_null_sub = process_json_file_and_calculate_pvalue(merge_sub_json_path, baseline_merge_sub_json_path)
            else:
                print(f"Warning: {merge_sub_json_path} or {baseline_merge_sub_json_path} does not exist.")
                continue

            # Add the results to the list
            results.append((nr_frames, layers, smooth, p_value, reject_null, p_value_sub, reject_null_sub))

    return results

# Example usage:
root_dir = 'eval_output'
model_name = 'LLaVA-NeXT-Video-7B-DPO'
#model_name = 'VideoLLaMA2-7B-16F'
results = process_directory_structure(root_dir, model_name)
save_ablation_results_to_csv(results)
