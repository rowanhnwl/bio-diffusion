import json
import os
import numpy as np

N_MOLECULES = 250

def get_dict_list(path_list):

    dict_list = []

    for path in path_list:
        with open(path, "r") as f:
            d = json.load(f)

        dict_list.append(d)

    return dict_list

def get_n_distinct(results_dict):
    for k1 in results_dict.keys():
        for k2 in results_dict[k1].keys():
            try:
                return results_dict[k1][k2]["n_distinct"]
            except:
                pass

def get_config_attributes(results_dict):
    # Return config name, constraint names
    config_name = list(results_dict.keys())[0]
    constraint_metrics = list(results_dict.values())[0]
    
    constraint_list = []
    for k in constraint_metrics.keys():
        if ":" not in k:
            constraint_list.append(k)

    return (config_name, constraint_list)

if __name__ == "__main__":
    
    eval_paths = [
        "task_arithmetic_eval/n1_no_constraint"
    ]

    out_path = "task_arithmetic_eval/summary/n3_no_constraint_summary.json"

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    full_eval_json_paths = [
        [os.path.join(ep, json_name) for json_name in sorted(os.listdir(ep))]
        for ep in eval_paths
    ]

    n_eval_files = len(full_eval_json_paths[0])

    out_dict_master = {}

    for i in range(n_eval_files):

        file_group = [results_group[i] for results_group in full_eval_json_paths]
        results_dicts_list = get_dict_list(file_group)

        # Assert that all files represent the same constraint or set of constraints
        assert len(set([os.path.basename(file_path) for file_path in file_group])) == 1, "Must compare the same constraints"

        config_name, constraints = get_config_attributes(results_dicts_list[0])
        out_dict_master[config_name] = {}

        for constraint in constraints:
            out_dict_master[config_name][constraint] = []

        if len(constraints) > 1:
            combined_constraint_name = ":".join(constraints)
            out_dict_master[config_name][combined_constraint_name] = []

        for results_dict in results_dicts_list:
            n_distinct = get_n_distinct(results_dict)
            
            for constraint in constraints:
                pass_rate = results_dict[config_name][constraint]["P(meets threshold)"]
                true_pass_rate = pass_rate * n_distinct / N_MOLECULES

                out_dict_master[config_name][constraint].append(true_pass_rate)

            if len(constraints) > 1:
                combined_pass_rate = results_dict[config_name][combined_constraint_name]
                true_combined_pass_rate = combined_pass_rate * n_distinct / N_MOLECULES

                out_dict_master[config_name][combined_constraint_name].append(true_combined_pass_rate)

        for k in list(out_dict_master[config_name].keys()):
            mean = np.mean(out_dict_master[config_name][k])
            std = np.std(out_dict_master[config_name][k])

            out_dict_master[config_name][k + "_mean"] = mean
            out_dict_master[config_name][k + "_std"] = std

    with open(out_path, "w") as f:
        json.dump(out_dict_master, f, indent=3)
