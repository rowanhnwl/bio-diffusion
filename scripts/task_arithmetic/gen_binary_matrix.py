import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import transformers
import torch
from torch import nn
# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
import numpy as np
import json
from tdc.single_pred import ADME
# import deepchem as dc
import random
from tqdm import tqdm
import os

def latent_space_constraint(dataset,input_latent_space_dict,constraint,constraint_val):
    # Inputs: 1) dataset (dictionary, with SMILES string/word string as key and property value as value)
    #         2) input_latent_space_dict (dictionary, with SMILES string/word string as key and latent space representation as value)
    #         3) constraint (string, property of interest)
    #         4) constraint_val (float, value of constraint to be mapped to the latent space)
    # Output: vector/matrix in the latent space corresponding to the constraint of interest

    # Step 1: Convert latent space representation into 1D tensor, standardizing size and stacking columns in matrix
    min_num_atoms = 10000 # arbitrary large number
    for latent_space in input_latent_space_dict.values():
        num_atoms = len(latent_space)
        if num_atoms < min_num_atoms:
            min_num_atoms = num_atoms
    #print(min_num_atoms)
    latent_space_tensor_dict = {}
    for smiles in input_latent_space_dict.keys():
        np_latent_space = np.array(input_latent_space_dict[smiles])
        latent_space_vec = np_latent_space[:min_num_atoms,0]
        for i in range(1,np_latent_space.shape[1]):
            latent_space_vec = np.hstack((latent_space_vec,np_latent_space[:min_num_atoms,i]))
        latent_space_vec = latent_space_vec.transpose()
        latent_space_tensor = torch.tensor(latent_space_vec)
        latent_space_tensor_dict[smiles] = latent_space_tensor
    # Step 2: Construct training and test splits
    # Random Split
    random.seed(0)
    smiles_list = list(dataset.keys())
    num_molecules = len(smiles_list)
    list_of_test_indices = random.sample(range(num_molecules),int(0.2*num_molecules))
    num_of_test_indices = len(list_of_test_indices)
    list_of_train_indices = [i for i in range(num_molecules) if i not in list_of_test_indices]
    num_of_train_indices = len(list_of_train_indices)
    batch_size = 8 # parameter that can be tuned

    train_latent_tensors = latent_space_tensor_dict[smiles_list[list_of_train_indices[0]]].unsqueeze(0)
    train_property_tensors = torch.tensor(dataset[smiles_list[list_of_train_indices[0]]]).unsqueeze(0).unsqueeze(0)
    for i in range(1,len(list_of_train_indices)):
        smiles = smiles_list[list_of_train_indices[i]]
        train_property_tensors_new = torch.vstack((train_property_tensors,torch.tensor(dataset[smiles_list[list_of_train_indices[i]]]).reshape(1).unsqueeze(0)))
        train_property_tensors = train_property_tensors_new
        train_latent_tensors = torch.vstack((train_latent_tensors,latent_space_tensor_dict[smiles_list[list_of_train_indices[i]]].unsqueeze(0)))
    # train_latent_tensors.reshape([num_of_train_indices,min_num_atoms,4])    
    test_latent_tensors = latent_space_tensor_dict[smiles_list[list_of_test_indices[0]]].unsqueeze(0)
    test_property_tensors = torch.tensor(dataset[smiles_list[list_of_test_indices[0]]]).reshape(1).unsqueeze(0)
    for i in range(1,len(list_of_test_indices)):
        smiles = smiles_list[list_of_test_indices[i]]
        test_property_tensors = torch.vstack((test_property_tensors,torch.tensor(dataset[smiles_list[list_of_test_indices[i]]]).reshape(1).unsqueeze(0)))
        test_latent_tensors = torch.vstack((test_latent_tensors,latent_space_tensor_dict[smiles_list[list_of_test_indices[i]]].unsqueeze(0)))
    # test_latent_tensors.reshape([num_of_test_indices,min_num_atoms,4])
        
    property_size = test_property_tensors.size(dim=1)
    latent_size = test_latent_tensors.size(dim=1)
    model = nn.Sequential(
        nn.Linear(property_size,latent_size//4),
        nn.SiLU(),
        nn.Linear(latent_size//4,latent_size//2),
        nn.SiLU(),
        nn.Linear(latent_size//2,latent_size),
    )
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(model.cuda().parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_data = torch.utils.data.TensorDataset(train_property_tensors, train_latent_tensors)
    test_data = torch.utils.data.TensorDataset(test_property_tensors, test_latent_tensors)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)
    
    # Step 3: Train model using training set, and test on test set

    def train(model, dataloader, optimizer, prev_updates):
        """
        Trains the model on the given data.
        
        Args:
            model (nn.Module): The model to train.
            dataloader (torch.utils.data.DataLoader): The data loader.
            loss_fn: The loss function.
            optimizer: The optimizer.
        """
        model.train()  # Set the model to training mode
        
        for batch_idx, (data, target) in enumerate(dataloader):
            n_upd = prev_updates + batch_idx
            
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            
            output = model(data)  # Forward pass
            loss_fn = nn.MSELoss()
            loss = loss_fn(output,target.float())
            # print(loss)
            loss.backward()
            
            if n_upd % 100 == 0:
                # Calculate and log gradient norms
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
            
               # print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f},  Grad: {total_norm:.4f}')
                
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
            
            optimizer.step()  # Update the model parameters
            
        return prev_updates + len(dataloader)

    def test(model, dataloader, cur_step):
        """
        Tests the model on the given data.
        
        Args:
            model (nn.Module): The model to test.
            dataloader (torch.utils.data.DataLoader): The data loader.
            cur_step (int): The current step.
        """
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(device)
                target = target.to(device)               
                output = model(data)  # Forward pass
                loss_fn = nn.MSELoss()
                test_loss = loss_fn(output,target.float())
            test_loss /= len(dataloader)
            #print(f'Test set loss: {test_loss}')
            test_loss_float = test_loss.detach().cpu()
            return test_loss_float.item()
    
    prev_updates = 0
    for epoch in range(num_epochs):
        #print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train(model, train_dataloader, optimizer, prev_updates)
        
    test_error = test(model, test_dataloader, prev_updates)
    
    # Step 4: Run model on model constraint to obtain vector in the latent space, then transform back to correct latent space
    constraint_tensor = torch.tensor([constraint_val]).unsqueeze(0)
    constraint_tensor = constraint_tensor.to(device)
    constraint_latent_tensor = model(constraint_tensor).to(device)
    constraint_latent_vec = constraint_latent_tensor.detach().cpu().numpy()
    constraint_latent_vec_origdim = np.zeros((min_num_atoms,np_latent_space.shape[1]))
    for i in range(constraint_latent_vec_origdim.shape[1]):
        for j in range(constraint_latent_vec_origdim.shape[0]):
            constraint_latent_vec_origdim[j,i] = constraint_latent_vec[0][i*min_num_atoms+j]
            # if i < np_latent_space.shape[1]-1:
            #     constraint_latent_vec_origdim[:,i] = constraint_latent_vec[i*min_num_atoms:(i+1)*min_num_atoms]
            # else:
            #     constraint_latent_vec_origdim[:,i] = constraint_latent_vec[i*min_num_atoms:]
    
    return constraint_latent_vec_origdim, test_error

def task_arithmetic(dataset_list,input_latent_space_dict_list,constraint_dict):
    # Inputs: 1) dataset_list (list of dataset dictionaries, each with SMILES string/word string as key and property value matrix as value)
    #         2) input_latent_space_dict_list (list of latent space dictionaries, each with SMILES string/word string as key and latent space representation as value)
    #         3) constraint_dict (dictionary, with constraint name as key and two values (constraint threshold, weight)
    # Output: vector/matrix in the latent space corresponding to the combination of constraints of interest
    
    # Step 1: Use previous function to calculate the "mapped" constraint in the latent space
    latent_vec_constraints_list = []
    test_errors_list = []
    constraint_weights = []
    for i in range(len(dataset_list)):
        # Get necessary inputs for latent space constraint mapping
        dataset = dataset_list[i]
        input_latent_space_dict = input_latent_space_dict_list[i]
        constraint = list(constraint_dict.keys())[i]
        constraint_val = constraint_dict[constraint]["threshold"]
        constraint_weights.append(constraint_dict[constraint]["weight"])
        # Obtain and store latent space vector for each constraint
        latent_space_constraint_vec, test_error = latent_space_constraint(dataset,input_latent_space_dict,constraint,constraint_val)
        latent_vec_constraints_list.append(latent_space_constraint_vec)
        test_errors_list.append(test_error)
    
    max_num_atoms = 0
    for vec in latent_vec_constraints_list:
        if vec.shape[0] > max_num_atoms:
            max_num_atoms = vec.shape[0]
    latent_vec_constraints_updated = []
    for vec in latent_vec_constraints_list:
        new_vec = np.zeros((max_num_atoms,vec.shape[1]))
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                new_vec[i,j] = vec[i,j]
        latent_vec_constraints_updated.append(new_vec)
    # Step 2: Use weights to add these vectors together
    norm_weights = [weight/sum(constraint_weights) for weight in constraint_weights]
    combined_latent_vec = np.zeros_like(latent_vec_constraints_updated[0])
    for i in range(len(latent_vec_constraints_updated)):
        combined_latent_vec += (norm_weights[i]*latent_vec_constraints_updated[i])
    
    return combined_latent_vec, test_errors_list

# TEST CASE 1B: Combination of two datasets
#constraint_dict_master = [{"Caco2 Permeability": {"threshold": -6.0,"weight": 1}},{"Lipophilicity": {"threshold": 2.0,"weight": 1}}, {"Solubility": {"threshold": 0.0, "weight": 1}}, {"Volume Distribution at Steady State": {"threshold": 5.0, "weight": 1}}, {"Acute Toxicity": {"threshold": 2.0, "weight": 1}}, {"TPSA": {"threshold": 50.0,"weight": 1}}, {"XLogP": {"threshold": -0.5, "weight": 1}}, {"Molecular Weight": {"threshold": 50.0, "weight": 1}}, {"Rotatable Bond Count": {"threshold": 1.0, "weight": 1}}]
#dataset_names_master = ["Caco2_Wang","Lipophilicity_AstraZeneca", "Solubility_AqSolDB", "LD50_Zhu", "VDss_Lombardo", "TPSA_pubchem", "XLogP_pubchem", "MolecularWeight_pubchem", "RotatableBondCount_pubchem"]

def generate_binary_matrix(constraint_dict_master, min_smiles_len, dataset_dir):

    constraint_names = [list(d.keys())[0] for d in constraint_dict_master]

    constraints_to_datasets_path = "eval_datasets/convert/constraint_to_dataset.json"
    with open(constraints_to_datasets_path, "r") as f:
        constraints_to_datasets = json.load(f)

    dataset_names_master = [constraints_to_datasets[c] for c in constraint_names]

    for i in range(len(dataset_names_master)):
        for j in range(len(dataset_names_master)):
            if i != j and ('pubchem' not in dataset_names_master[i] or 'pubchem' not in dataset_names_master[j]):

                constraint_name_1 = list(constraint_dict_master[i].keys())[0]
                constraint_name_2 = list(constraint_dict_master[j].keys())[0]

                print("GENERATING THE TASK ARITHMETIC MATRIX")
                print('Constraint 1: ', constraint_name_1)
                print('Constraint 2: ', constraint_name_2)
                constraint_dict = constraint_dict_master[i].copy()
                for key, value in constraint_dict_master[j].items():
                    constraint_dict[key] = value
                #print(constraint_dict)
                dataset_names = [dataset_names_master[i],dataset_names_master[j]]

                dataset_dicts = []
                input_latent_space_dicts = []
                bad_smi_dict = {}

                with open(f'{dataset_dir}/latent/updated_pubchem_latent_space_dict.json','r') as pubchem_latent_json:
                    pubchem_latent_dict = json.load(pubchem_latent_json)
                for i in range(len(dataset_names)):
                    dataset_name = dataset_names[i]
                    if 'pubchem' in dataset_name:
                        with open(f'{dataset_dir}/updated_{dataset_name}_new_dataset.json','r') as d:
                            dataset_dict = json.load(d)
                        list_of_bad_smi = []
                        for smi in dataset_dict.keys():
                            if dataset_dict[smi][0] == {"CID": 0}:
                                list_of_bad_smi.append(smi)
                            elif smi not in pubchem_latent_dict.keys():
                                list_of_bad_smi.append(smi)
                            elif dataset_name.split('_')[0] not in dataset_dict[smi][0].keys():
                                list_of_bad_smi.append(smi)
                            elif len(smi) < min_smiles_len: # get rid of molecules that are too small
                                list_of_bad_smi.append(smi)
                        for smi in list_of_bad_smi:
                            del dataset_dict[smi]
                        fixed_dataset = {}
                        for smi in dataset_dict.keys():
                            prop_val = dataset_dict[smi][0][dataset_name.split('_')[0]]
                            fixed_dataset[smi] = float(prop_val)
                        fixed_dataset_json = json.dumps(fixed_dataset)
                        with open(f'{dataset_dir}/{dataset_name}_fixed_dataset.json','w') as d:
                            d.write(fixed_dataset_json)
                        dataset_dicts.append(f'{dataset_dir}/{dataset_name}_fixed_dataset.json')
                        input_latent_space_dicts.append(f'{dataset_dir}/latent/updated_pubchem_latent_space_dict.json')
                        bad_smi_dict[f'{dataset_dir}/latent/updated_pubchem_latent_space_dict.json'] = list_of_bad_smi
                    else:
                        with open(f'{dataset_dir}/updated_{dataset_name}_new_dataset.json','r') as d:
                            dataset_dict = json.load(d)
                        list_of_bad_smi = []
                        for smi in dataset_dict.keys():
                            if len(smi) < min_smiles_len:
                                list_of_bad_smi.append(smi)
                        for smi in list_of_bad_smi:
                            del dataset_dict[smi]
                        fixed_dataset_json = json.dumps(dataset_dict)
                        with open(f'{dataset_dir}/{dataset_name}_fixed_dataset.json','w') as d:
                            d.write(fixed_dataset_json)
                        input_latent_space_dicts.append(f'{dataset_dir}/latent/updated_{dataset_name}_latent_space_dict.json')
                        dataset_dicts.append(f'{dataset_dir}/{dataset_name}_fixed_dataset.json')
                        bad_smi_dict[f'{dataset_dir}/latent/updated_{dataset_name}_latent_space_dict.json'] = list_of_bad_smi
                datasets = []
                input_latent_spaces = []
                for dataset_filename in dataset_dicts:
                    with open(dataset_filename,'r') as d:
                        dataset_dict = json.load(d)
                    datasets.append(dataset_dict)
                for latent_filename in input_latent_space_dicts:
                    with open(latent_filename,'r') as l:
                        latent_dict = json.load(l)
                    list_of_bad_smi = bad_smi_dict[latent_filename]
                    for smi in list_of_bad_smi:
                        if smi in latent_dict.keys():
                            del latent_dict[smi]
                    input_latent_spaces.append(latent_dict)
                combined_latent_vec, test_errors_list = task_arithmetic(datasets,input_latent_spaces,constraint_dict)        

                return combined_latent_vec.tolist()