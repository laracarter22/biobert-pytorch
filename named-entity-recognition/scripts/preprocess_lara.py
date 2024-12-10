#Step 1 Load .jsonl data

import json

def load_jsonl_data(json_path):
    texts = []
    entities = []
    with open(json_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'])
            entities.append([(int(ent['start']), int(ent['end']), ent['label']) for ent in data['spans']])
    return texts, entities

#Step 2: Tokenize and align labels

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

def tokenize_and_align_labels(texts, entities, max_len=512):
    tokenized_inputs = tokenizer(texts, padding=False, truncation=True, max_length=max_len, return_offsets_mapping=True)

    labels = []
    for i, (text, entity_list) in enumerate(zip(texts, entities)):
        offsets = tokenized_inputs['offset_mapping'][i]
        label_list = ['O'] * len(offsets)  # 'O' for non-entity tokens
        for start, end, label in entity_list:
            for j, (start_offset, end_offset) in enumerate(offsets):
                if start_offset < end and end_offset > start:
                    label_list[j] = 'B-PK' if label_list[j] == 'O' else 'I-PK'
        
        labels.append(label_list)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


#Step 3: Convert to .tsv

import pandas as pd

def create_tsv(tokenized_data, out_path, split_name):
    rows = []
    for idx, (input_ids, labels) in enumerate(zip(tokenized_data['input_ids'], tokenized_data['labels'])):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Write token-label pairs
        for token, label in zip(tokens, labels):
            rows.append({
                "uid": str(idx),
                "token": token,
                "label": label
            })
    
    # Convert rows to DataFrame and save as .tsv
    df = pd.DataFrame(rows)
    tsv_path = f"{out_path}/{split_name}.tsv"
    df.to_csv(tsv_path, sep='\t', index=False, header=False)
    print(f"TSV file saved to {tsv_path}")

#Step 4: Convert .tsv to .conll

def tsv_to_conll(tsv_path, conll_path):
    tsv_data = pd.read_csv(tsv_path, sep='\t', names=["uid", "token", "label"])
    
    with open(conll_path, 'w', encoding='utf-8') as conll_file:
        for _, row in tsv_data.iterrows():
            token = row['token']
            label = row['label']
            conll_file.write(f"{token} {label}\n")
        
        conll_file.write("\n")  # Add a blank line between sentences


#Overall processing for preprocess.py

def process_data(jsonl_path, tsv_out_path, conll_out_path):
    # Load data from jsonl file
    texts, entities = load_jsonl_data(jsonl_path)

    # Tokenize and align labels
    tokenized_data = tokenize_and_align_labels(texts, entities)

    # Convert to TSV format
    create_tsv(tokenized_data, tsv_out_path, 'data')  # Adjust 'data' to train/dev/test as needed

    # Convert to CoNLL format
    tsv_to_conll(f"{tsv_out_path}/data.tsv", f"{conll_out_path}/data.conll")

if __name__ == "__main__":
    jsonl_path = './pk_dataset'  # Path to your JSONL file
    tsv_out_path = './lara_preprocessed_tsv'  # Path to save the TSV file
    conll_out_path = './lara_preprocessed_conll'  # Path to save the CoNLL file

    # Process the data
    process_data(jsonl_path, tsv_out_path, conll_out_path)