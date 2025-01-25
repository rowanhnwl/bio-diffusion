import json
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    results_path = args.path
    out_path = args.out

    results_jsons = os.listdir(results_path)

    out_dict = {}

    for filename in results_jsons:
        json_path = os.path.join(results_path, filename)

        with open(json_path, "r") as f:
            eval_results = json.load(f)

        param_eval_results = list(eval_results.items())

        constraint_name = list(param_eval_results[0][1].keys())[0]

        if list(param_eval_results[0][1][constraint_name].keys())[-1] == "n_good":
            # Sort based on n_good
            param_eval_results = sorted(
                param_eval_results,
                key=lambda x: x[1][constraint_name]["n_good"],
                reverse=True
            )
        else:

            # Sort based on (pass_rate) * (n_valid)
            param_eval_results = sorted(
                param_eval_results,
                key=lambda x: (x[1][constraint_name]["pass_rate"]) * (x[1][constraint_name]["n_valid"]),
                reverse=True
            )

        for i, param_eval_result in enumerate(param_eval_results):
            config_name = param_eval_result[0]
            # Remove the constraint name
            config_name_list = config_name.split("_")
            config_name_list.pop(0)
            param_set = "_".join(config_name_list)

            if config_name == "no-constraint":
                param_set = "no-constraint"

            if list(param_eval_results[0][1][constraint_name].keys())[-1] == "n_good":
                try:
                    out_dict[param_set][constraint_name] = param_eval_result[1][constraint_name]["n_good"]
                except:
                    out_dict[param_set] = {}
                    out_dict[param_set][constraint_name] = param_eval_result[1][constraint_name]["n_good"]
            else:
                try:
                    out_dict[param_set][constraint_name] = (param_eval_result[1][constraint_name]["pass_rate"]) * (param_eval_result[1][constraint_name]["n_valid"])
                except:
                    out_dict[param_set] = {}
                    out_dict[param_set][constraint_name] = (param_eval_result[1][constraint_name]["pass_rate"]) * (param_eval_result[1][constraint_name]["n_valid"])
    # Sort the out_dict by average ranking
    out_dict_list = list(out_dict.items())
    out_dict_means = [(x[0], np.mean(list(x[1].values()))) for x in out_dict_list]

    for x in out_dict_means:
        out_dict[x[0]]["Mean good"] = x[1]

    out_dict_sortable = list(out_dict.items())
    out_dict_sorted = sorted(
        out_dict_sortable,
        key=lambda x: x[1]["Mean good"],
        reverse=True
    )

    out_dict_final = {}
    for x in out_dict_sorted:
        out_dict_final[x[0]] = x[1]

    with open(out_path, "w") as f:
        json.dump(out_dict_final, f, indent=3)
