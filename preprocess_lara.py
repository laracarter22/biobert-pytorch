import json
import os
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import ast
import re
from multi_exp_def import MultiTaskDefs

def load_jsonl_data(json_path):
    texts = []
    entities = []
    # Mount Google Drive if necessary (for Google Colab)

    #open file in read mode (r)
    #iterates over each line
    #extracts data from each JSON object

    with open(json_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data['text'])
            entities.append([(int(ent['start']), int(ent['end']), ent['label']) for ent in data['spans']])
    return texts, entities

#I think I'm still missing tasks... 
#TASKS NEED TO BE WORKED ON 

def load_tsv_data(tsv_path, task):
    tsv_data = pd.read_csv(tsv_path, sep='\t',
                           names=["uid", "bcls_label", "mcls_label", "mt_label", "ner_label", "token_id"])
    

    if(task == 'Joint-three-mt-b'):
        tsv_data.insert(1, 'cls_label', tsv_data['bcls_label'])
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label'])

    if(task == 'Joint-three-mt-m'):
        tsv_data.insert(1, 'cls_label', tsv_data['mcls_label'])
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label'])


    #ner and b
    if(task == 'Joint-two-b'):
        tsv_data.insert(1, 'cls_label', tsv_data['bcls_label'])
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label', 'mt_label'])

    #ner and m
    if(task == 'Joint-two-m'):
        tsv_data.insert(1, 'cls_label', tsv_data['mcls_label'])
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label', 'mt_label'])

    #b and mt
    if(task == 'Joint-bCLS-mtCLS'):
        tsv_data.insert(1, 'cls_label', tsv_data['bcls_label'])
        tsv_data['ner_label'] = tsv_data['mt_label']
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label', 'mt_label'])

    #m and mt
    if(task == 'Joint-mCLS-mtCLS'):
        tsv_data.insert(1, 'cls_label', tsv_data['mcls_label'])
        tsv_data['ner_label'] = tsv_data['mt_label']
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label', 'mt_label'])

    #b and m
    if(task == 'Joint-bCLS-mCLS'):
        tsv_data = tsv_data.drop(columns=['ner_label', 'mt_label'])

    #ner and mt
    if(task == 'Joint-mt'):
        tsv_data = tsv_data.drop(columns=['bcls_label', 'mcls_label'])
    
    return tsv_data

#Now unused, but keep just incase
def create_vocab(vocab_file):
    trainTexts, entities = load_jsonl_data(os.path.join(data_path, "train.jsonl"))
    testTexts, entities = load_jsonl_data(os.path.join(data_path, "test.jsonl"))
    devTexts, entities = load_jsonl_data(os.path.join(data_path, "dev.jsonl"))
    
    #concatenate
    texts = trainTexts + testTexts + devTexts

    # Step 1: Tokenize the sentences and collect unique words
    unique_words = set()  # Using a set to store unique words

    for sentence in texts:
        #words = re.split(r'(?<=[.!?]) +', sentence)
        words = sentence.split()  # Split sentence into words
        unique_words.update(words)  # Add words to the set (duplicates automatically handled)

    # Step 2: (Optional) Sort the unique words
    sorted_unique_words = sorted(unique_words)

    # Step 3: Write the unique words to vocab.txt
    with open('vocab.txt', 'w', encoding='utf-8') as vocab_file:
        for word in sorted_unique_words:
            vocab_file.write(word + '\n')

#model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
#tokenizer = BertWordPieceTokenizer(vocab=vocab_path, lowercase=False)

#ADDING THE VOCABULARY
vocab = tokenizer.get_vocab()

def whitespace_tokenizer(text):
    """
    Tokenizes a sentence based on whitespace.
    
    Args:
    text (str): The input sentence to tokenize.

    Returns:
    list: A list of tokens (words).
    """
    return text.split()

#ADDED FROM HERE 
def handle_oov_words(text, vocab):
    """
    This function checks if a word exists in the vocabulary and avoids splitting OOV words.

    Args:
        text (str): The input text to preprocess.
        vocab (dict): The vocabulary dictionary.

    Returns:
        list: A list of tokens where OOV words are not split.
    """
    tokens = text.split()  # Split by spaces (words)
    new_tokens = []

    for token in tokens:
        # Check if token is in the vocab, if not, treat it as a whole (OOV word handling)
        if token in vocab:
            new_tokens.append(token)  # Keep the token as is
        else:
            # For OOV words, you can use a placeholder or just append the whole word
            new_tokens.append(token)  # This prevents subword splitting for OOV words

    return new_tokens

#TO HERE

def tokenize_and_align_labels(texts, entities,  max_len=512):
    tokenized_inputs = tokenizer(texts, padding=False, truncation=True, max_length=max_len, return_offsets_mapping=True, is_split_into_words=False)

    labels = []
    new_label_column = []

    for i, (text, entity_list) in enumerate(zip(texts, entities)):

        #ADDED THIS HERE
        processed_text = handle_oov_words(text, vocab)

        offsets = tokenized_inputs['offset_mapping'][i]
        label_list = ['O'] * len(offsets)
        new_label_type = ['O'] * len(offsets)

        for start, end, label in entity_list:
            entity_started = False
            for j, (start_offset, end_offset) in enumerate(offsets):
                if start_offset < end and end_offset > start: 
                    if not entity_started:
                        label_list[j] = 'B-PK'
                        entity_started = True
                    else:
                        label_list[j] = 'I-PK'
                #if start_offset <= start < end_offset or start_offset < end <= end_offset:
                 #   if start_offset == start:
                  #      label_list[j] = 'B-PK'
                   # else:
                    #    label_list[j] = 'I-PK'
        
        #new_label_type = ['O'] * len(label_list)

        for j in range(len(label_list)):
            if label_list[j] == 'B-PK':
                if j + 1 < len(label_list) and label_list[j + 1] == 'I-PK':
                    new_label_type[j] = 'B-Multi'
                else:
                    new_label_type[j] = 'B-Single'
            elif label_list[j] == 'I-PK':
                #if j > 0 and label_list[j - 1] == 'B-PK':
                    new_label_type[j] = 'I-Multi'
                #elif j + 1 < len(label_list) and label_list[j + 1] == 'I-PK':
                   # new_label_type[j] = 'I-Multi'
    
        # Truncate all to the minimum length

        min_len = min(len(label_list), len(new_label_type), len(tokenized_inputs['input_ids'][i]))
        label_list = label_list[:min_len]
        new_label_type = new_label_type[:min_len]
        tokenized_inputs['input_ids'][i] = tokenized_inputs['input_ids'][i][:min_len]

        labels.append(label_list)
        new_label_column.append(new_label_type)
    
    tokenized_inputs["labels"] = labels
    tokenized_inputs["label_type"] = new_label_column
    return tokenized_inputs

def create_all_tsvs(data_path, out_path):
    create_tsv(data_path, out_path, 'dev')
    create_tsv(data_path, out_path, 'test')
    create_tsv(data_path, out_path, 'train')

def create_tsv(dataset_folder_path, out_path, data_name):
    print(f"Creating tsv {data_name}.tsv")
    dataset_path = os.path.join(dataset_folder_path, data_name + ".jsonl")
    texts, entities = load_jsonl_data(dataset_path)
    tokenized_dataset = tokenize_and_align_labels(texts, entities)

    #token_to_id = {token: idx for idx, token in enumerate(set(t for sublist in tokenized_dataset['input_ids'] for t in sublist))}

    rows = []
    for idx, (input_ids, labels, label_types) in enumerate(zip(tokenized_dataset['input_ids'], tokenized_dataset['labels'], tokenized_dataset['label_type'])):
        has_px_label = any(label != 'O' for label in labels)
        px_label_count = min(3, sum(1 for label in labels if label == 'B-PK'))

        # Get the tokens from the input_ids
        #token_ids = [token_to_id[token] for token in input_ids].
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        rows.append({
            "uid": str(idx),
            "bcls_label": int(has_px_label),
            "mcls_label": px_label_count,
            "labels": labels,
            "label_type": label_types,
            "tokens": tokens,
        })

    df = pd.DataFrame(rows)

    # Save the DataFrame as a .tsv file
    out_file_path = os.path.join(out_path, data_name + '.tsv')
    df.to_csv(out_file_path, sep='\t', index=False, header=False)

def parse_list_column(source):
    if isinstance(source, str):
        try:
            return ast.literal_eval(source)
        except (ValueError, SyntaxError):
            print(f"Error parsing {source}. Returning as is.")
            return source
    return source

# Define mapping functions
def map_mt_labels(ls):
    ls = ast.literal_eval(ls)  # Convert from string to list
    mapping = {'O': 0, 'B-PK': 1, 'I-PK': 2}
    return [mapping.get(item, 0) for item in ls]  # Ensure list of integers

def map_ner_labels(ls):
    # ls = ast.literal_eval(source)  # Convert from string to list
    ls = parse_list_column(ls)
    mapping = {'O': 0, 'B-Single': 1, 'B-Multi': 2, 'I-Multi': 3}
    return [mapping.get(item, 0) for item in ls] # Ensure list of integers

def map_tokens(source):
    #tokens = ast.literal_eval(source)  # Convert from string to list
    tokens = parse_list_column(source)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids  # Convert tokens to IDs

def tsv_to_json(tsv_path, json_path, task):
    tsv_data = load_tsv_data(tsv_path, task)

    tsv_data['uid'] = tsv_data['uid'].astype(str)

    if 'mt_label' in tsv_data.columns:
        tsv_data['mt_label'] = tsv_data['mt_label'].apply(map_mt_labels)    

    if 'ner_label' in tsv_data.columns:
        tsv_data['ner_label'] = tsv_data['ner_label'].apply(map_ner_labels)

    # Apply the mapping functions
    tsv_data['token_id'] = tsv_data['token_id'].apply(map_tokens)
    
        
    if(task == 'Joint-bCLS-mCLS'):
        tsv_data['type_id'] = tsv_data['bcls_label'].apply(lambda x: [0])
    else:
        # Add type_id with the same length as ner_label and filled with 0s
        tsv_data['type_id'] = tsv_data['ner_label'].apply(lambda x: [0] * len(x))

    # Convert DataFrame to JSON format
    json_records = tsv_data.to_dict(orient="records")

    # Save JSON lines to file
    with open(json_path, 'w') as json_file:
        for record in json_records:
            json.dump(record, json_file)
            json_file.write('\n')

    print(f"JSON file saved to {json_path}")
    

#OKAY NEED TO WORK ON THE CREATION OF TASK DATA

def create_each_task_data(data_path, tsv_path, json_path, yml_path):
    task_defs = MultiTaskDefs(yml_path)

    tasks = task_defs.task

    for task in tasks:
        data_format = task_defs.data_format_map[task]   #holds TwoPremiseAndTwoSequence, or something
        split_names = task_defs.split_names_map[task]   #holds [train, test, dev]

        if task not in task_defs.task_def_dic:
            raise KeyError('%s: Cannot process this task' % task)
        
        for split_name in split_names:
            split_tsv_path = f'{tsv_path}/{split_name}.tsv'
            split_json_path = f'{json_path}/{task}_{split_name}.json'
            tsv_to_json(split_tsv_path, split_json_path, task)
    


if __name__ == '__main__':
    data_path = './dataset'
    tsv_path = './task_def_joint/canonical_data'
    json_path = './task_def_joint/canonical_data/bert_cased'
    yml_path = './task_def_joint/canonical_data/multi_task_def.yml'

    #Now using AutoTokenizer so dont need this anymore
    #create_vocab(vocab_path)

    create_all_tsvs(data_path, tsv_path)
    #convert_tsvs_to_json(tsv_path, json_path)

    create_each_task_data(data_path, tsv_path, json_path, yml_path)