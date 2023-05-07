import json
import copy
import pandas as pd
import torch
import numpy as np
from math import log
from tqdm import tqdm
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from transformers import T5TokenizerFast, T5ForConditionalGeneration


def two_segment_baseline(sampled_data):
    all_results = []
    for key in list(sampled_data.keys()):
        length = len([item for sublist in sampled_data[key] for item in sublist])
        all_results.append([int(length/2)-1])
    print('Log: 2 average-length segments baseline done')
    return all_results


def three_segment_baseline(sampled_data):
    all_results = []
    for key in list(sampled_data.keys()):
        length = len([item for sublist in sampled_data[key] for item in sublist])
        all_results.append([int((length)/3-1),int((length)*2/3-1)])
    print('Log: 3 average-length segments baseline done')
    return all_results


def five_segment_baseline(sampled_data):
    all_results = []
    for key in list(sampled_data.keys()):
        length = len([item for sublist in sampled_data[key] for item in sublist])
        all_results.append([int((length)/5-1),int((length)*2/5-1),int((length)*3/5-1),int((length)*4/5-1)])

    print('Log: 5 average-length segments baseline done')
    return all_results


def nsp(sampled_data):
    print('Log: Probility calculation using BERT NSP started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    all_probs = []
    all_seq_A = []
    all_seq_B = []
    all_seq_idx = [0] # index to separate probabilities from difference sentences

    prev_id = 0
    for id in sent_idx:
        sents = all_sentences[prev_id:id]
        seq_A = sents[:-1]
        seq_B = sents[1:]
        all_seq_A = all_seq_A + seq_A
        all_seq_B = all_seq_B + seq_B
        all_seq_idx.append(all_seq_idx[-1]+len(seq_A))
        prev_id = id
    all_seq_idx = all_seq_idx[1:]
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    for i in tqdm(range(len(all_seq_A))):
        encoded = tokenizer.encode_plus(all_seq_A[i], text_pair=all_seq_B[i], return_tensors='pt')
        seq_relationship_logits = model(**encoded)[0]
        prob = softmax(seq_relationship_logits, dim=1).tolist()[0][0]
        all_probs.append(prob)
    
    all_results = []
    all_results2 = []
    prev_id = 0
    for id in all_seq_idx:
        probs = all_probs[prev_id:id]
        probs = probs[1:-1] # first or last element cannot be a section by itself
        result = [probs.index(sorted(probs)[0])+1] # 2 sections
        result2 = sorted([probs.index(sorted(probs)[0])+1,probs.index(sorted(probs)[1])+1]) # 3 sections
        all_results.append(result)
        all_results2.append(result2)
        prev_id = id
    print('Log: Probility calculation using BERT NSP done')
    return all_results, all_results2


def sbert_cluster(sampled_data):
    print('Log: Segmentation with Sbert Clustering started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    all_embeddings = sbert_model.encode(all_sentences)

    prev_id = 0
    for id in tqdm(sent_idx):
        result = []
        embed = all_embeddings[prev_id:id]
        idx_s = 0
        for idx_e in range(len(embed)-1):
            if idx_e < idx_s+2:
                continue
            res = util.community_detection(embed[idx_s:idx_e], min_community_size=1, threshold=0.46)
            if len(res) >= 2:
                result.append(idx_e-1)#-2
                idx_s = idx_e-1
        prev_id = id
        all_results.append(result)
    print('Log: Segmentation with Sbert Clustering done')        
    return all_results


def agglomerative(sampled_data):
    print('Log: Segmentation with Agglomerative Clustering started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_embeddings = embedder.encode(all_sentences)
    all_embeddings = all_embeddings/np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    prev_id = 0
    for id in tqdm(sent_idx):
        result = []
        r = []
        pro_result = []
        embed = all_embeddings[prev_id:id]
        embed = embed /  np.linalg.norm(embed, axis=1, keepdims=True)

        clustering_model = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.6) #, linkage='ward', distance_threshold=1.2
        clustering_model.fit(embed)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            clustered_sentences[cluster_id].append(1)      

        for i, cluster in clustered_sentences.items():
            r.append(len(cluster))
        for i in range(len(r)):
            if i == 0:
                result.append(r[i]-1)
            else:
                result.append(result[i-1]+r[i])
        for i in range(len(result[:-1])):
            if result[i] <= 2:
                continue
            if i > 0 :
                if result[i]-result[i-1] <=1:
                    continue
            pro_result.append(result[i])
        all_results.append(pro_result)
        prev_id = id
    print('Log: Segmentation with Agglomerative Clustering done')        
    return all_results


def get_sub_table(table, row_start , row_end, col_start, col_end):
    sub_table = []
    for i in range(row_start, row_end+1):
        sub_row = table[i][col_start:col_end+1]
        sub_table.append(sub_row)
    return sub_table


def get_best_split_point(table):
    f_splits = []
    e_splits = []
    overall_perplexity = table[len(table)-1][0]
    for i in range(len(table)-1):
        if table[i][0] < overall_perplexity:
            f_splits.append(i)
        if table[len(table)-1][i] < overall_perplexity:
            e_splits.append(i-1)
    if len(f_splits) == 0 and len(e_splits): # there is no split point
        return 0
    else: # there is a split point
        best_split = 0
        best_score = float('inf')
        splits = list(set(f_splits+e_splits))
        for split in splits:
            front_score = table[split][0]
            end_score = table[len(table)-1][split+1]
            score = front_score + end_score # summation of Cross Entropy Score
            if score < best_score:
                best_split = split
        return best_split


def perplexity(sampled_data):
    print('Log: Segmentation with low perplexity started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    device = "cuda"
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    prev_id = 0
    for id in tqdm(sent_idx):
        result = []
        table = [] # to store all perplexities in a 2D list structure as a table
        sent = all_sentences[prev_id:id]
        for i in range(len(sent)):
            row = [] # one row in the table
            for j in range(i):
                embed = tokenizer("\n\n".join(sent[j:i+1]), return_tensors="pt")
                trg_len = embed.input_ids[:, 0:1024].size(1)-1
                input_ids = embed.input_ids[:, 0:1024].to(device)
                target_ids = input_ids.clone()
                #target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0]
                ppl = torch.exp(neg_log_likelihood*trg_len/(trg_len+1))
                row.append(ppl.item())
            while len(row) < len(sent):
                row.append(float('inf'))
            table.append(row)
        '''
        To Manipulate with the perplexity table here. Need much adjusting.
        '''
        split = get_best_split_point(table) # 0 if no split point
        if split:
            front_split = None
            end_split = None
            if split > len(table)-1-split:
                front_sub_table = get_sub_table(table, 0, split, 0, split)
                front_split = get_best_split_point(front_sub_table)
            else:
                end_sub_table = get_sub_table(table, split+1, len(table)-1, split+1, len(table)-1)
                end_split = get_best_split_point(end_sub_table)

            if front_split:
                result.append(front_split)
            result.append(split)
            if end_split:
                result.append(end_split+split+1)
        all_results.append(result)
        prev_id = id
    print('Log: Segmentation with low perplexity done')        
    return all_results


def get_slide_spilt(table):
    f_threshold = 10000 # difference of front perplexity that is counted
    b_threshold = 10000 # difference of back perplexity that is counted
    for i in range(1, len(table)-1):
        front_prev = table[i][0]
        front_next = table[i+1][0]
        back_prev = table[len(table)-1][i]
        back_next = table[len(table)-1][i+1]
        if (front_next - front_prev > 0.2 and back_prev-back_next > 0.2) or front_next - front_prev > f_threshold or back_prev - back_next > b_threshold: # perplexity increase for front and drop for end 
            return i
    return 0


def perplexity2(sampled_data):
    print('Log: Segmentation with sliding perplexity started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    device = "cuda"
    model_id = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    prev_id = 0
    for id in tqdm(sent_idx):
        result = []
        table = [] # to store all perplexities in a 2D list structure as a table
        sent = all_sentences[prev_id:id]
        for i in range(len(sent)):
            row = [] # one row in the table
            for j in range(i):
                embed = tokenizer("\n\n".join(sent[j:i+1]), return_tensors="pt")
                trg_len = embed.input_ids[:, 0:1024].size(1)-1
                input_ids = embed.input_ids[:, 0:1024].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0]
                ppl = torch.exp(neg_log_likelihood*trg_len/(trg_len+1))
                row.append(ppl.item())
            while len(row) < len(sent):
                row.append(float('inf'))
            table.append(row)
        split = get_slide_spilt(table)
        prev_split = -1
        while split:
            result.append(prev_split+1+split)
            subtable = get_sub_table(table, split+1, len(table)-1, split+1, len(table)-1)
            table = subtable # update table to the smaller one
            prev_split = split
            split = get_slide_spilt(subtable)
        all_results.append(result)
        prev_id = id
    print('Log: Segmentation with minimizing perplexity done')        
    return all_results


def perplexity2_t5(sampled_data):
    print('Log: Segmentation with sliding perplexity started')
    all_results = []
    all_sentences = []
    sent_idx = [0]
    for key in list(sampled_data.keys()):
        all_sentences = all_sentences + [item for sublist in sampled_data[key] for item in sublist]
        sent_idx.append(sent_idx[-1]+len([item for sublist in sampled_data[key] for item in sublist]))
    sent_idx = sent_idx[1:] # remove the zero

    device = "cuda"
    model_id = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(model_id)

    prev_id = 0
    for id in tqdm(sent_idx):
        result = []
        table = [] # to store all perplexities in a 2D list structure as a table
        sent = all_sentences[prev_id:id]
        for i in range(len(sent)):
            row = [] # one row in the table
            for j in range(i):
                embed = tokenizer("\n\n".join(sent[j:i+1]), return_tensors="pt")
                trg_len = embed.input_ids[:, 0:1024].size(1)-1
                input_ids = embed.input_ids[:, 0:1024].to(device)
                target_ids = input_ids.clone()
                #target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0]
                ppl = torch.exp(neg_log_likelihood*trg_len/(trg_len+1))
                row.append(ppl.item())
            while len(row) < len(sent):
                row.append(float('inf'))
            table.append(row)
        split = get_slide_spilt(table)
        if split:
            result.append(split)
            subtable = get_sub_table(table, split+1, len(table)-1, split+1, len(table)-1)
            split2 = get_slide_spilt(subtable)
            if split2:
                result.append(split+1+split2)
        all_results.append(result)
        prev_id = id
    print('Log: Segmentation with minimizing perplexity done')        
    return all_results



def edit_distance(results_original, keys_original, evaluate_size, PENALTY):
    results = copy.deepcopy(results_original) #copy
    keys = copy.deepcopy(keys_original) #copy
    sum_dist = 0
    for i in range(len(results)):
        result = results[i]
        key = keys[i]
        dist = 0
        if len(result) == len(key):
            for i in range(len(result)):
                dist = dist + abs(result[i]-key[i])
        elif len(result) < len(key):
            dist = dist + (len(key)-len(result))*PENALTY
            for r in result:
                if r in key:
                    result.remove(r)
                    key.remove(r)
            for r in result: 
                r_close = min(key, key=lambda x:abs(x-r))
                dist = dist + abs(r - r_close)
                key.remove(r_close)
        else: # more segments than the key
            dist = dist + (len(result)-len(key))*PENALTY
            for r in result:
                if r in key:
                    result.remove(r)
                    key.remove(r)
            for k in key: 
                k_close = min(result, key=lambda x:abs(x-k))
                dist = dist + abs(k - k_close)
                result.remove(k_close)
        sum_dist = sum_dist + dist
    return sum_dist/evaluate_size


def get_answer_segments(sampled_data):
    all_segments = []
    for key in tqdm(list(sampled_data.keys())):
        data = sampled_data[key]
        if len(data) == 1: # single section
            all_segments.append([])
        else: # multiple sections
            segments = []
            segment_pos = -1
            for sec in data:
                segment_pos = segment_pos + len(sec)
                segments.append(segment_pos)
            segments = segments[:-1]
            all_segments.append(segments)
    print('Log: Answer segmentation split position conversion done')
    return all_segments


def print_result(method_name, all_results, all_segments, evaluate_size):
    avg_dist3 = edit_distance(all_results, all_segments, evaluate_size, PENALTY=3)
    avg_dist4 = edit_distance(all_results, all_segments, evaluate_size, PENALTY=4)
    print("Using the method of "+ method_name +", for "+str(evaluate_size)+ " samples, the average edit distance is: "+str(avg_dist2)+", "+str(avg_dist3)+", "+str(avg_dist4))


if __name__ == "__main__":

    path = "../data/segmentation/wikiHow_segmentation_data.json"
    with open(path) as f:
        data = json.load(f)

    all_size = len(data)
    evaluate_size = all_size
    sampled_data = dict((k, data[k]) for k in list(data.keys())[:evaluate_size])

    '''
    Calculate the segmentation with different methods
    '''
    all_segments = get_answer_segments(sampled_data)
    all_results_two_segment = two_segment_baseline(sampled_data)
    all_results_three_segment = three_segment_baseline(sampled_data)
    all_results_five_segment = five_segment_baseline(sampled_data)
    all_results_nsp1, all_results_nsp2 = nsp(sampled_data)
    all_results_sbert = sbert_cluster(sampled_data) #topic detection
    all_results_agglo = agglomerative(sampled_data)
    all_results_perplex = perplexity(sampled_data)
    all_results_perplex2 = perplexity2(sampled_data)
    all_results_perplex2_t5 = perplexity2_t5(sampled_data)

    print_result("2 average-length segments (baseline)", all_results_two_segment, all_segments, evaluate_size)
    print_result("3 average-length segments (baseline)", all_results_three_segment, all_segments, evaluate_size)
    print_result("5 average-length segments (baseline)", all_results_five_segment, all_segments, evaluate_size)
    print_result("NSP (2 sections)", all_results_nsp1, all_segments, evaluate_size)
    print_result("NSP (3 sections)", all_results_nsp2, all_segments, evaluate_size)
    print_result("Sbert Fast Clustering", all_results_sbert, all_segments, evaluate_size) # topic detection
    print_result("Agglomerative Clustering", all_results_agglo, all_segments, evaluate_size)
    print_result("Finding low perplexity", all_results_perplex, all_segments, evaluate_size)
    print_result("Sliding perplexity", all_results_perplex2, all_segments, evaluate_size)
    print_result("Sliding perplexity t5", all_results_perplex2_t5, all_segments, evaluate_size)

    #Split the flat text into sections with our own segmentation
    segments = []
    goals = []
    for i in tqdm(range(len(all_results_three_segment))):
        result = all_results_three_segment[i]
        goal = list(sampled_data.keys())[i]
        texts = [item for sublist in sampled_data[goal] for item in sublist]
        real_goal = goal
        prev = 0
        for i in range(len(result)):
            text = " ".join(texts[prev:result[i]+1])
            if text!='':
                segments.append(text)
                goals.append(real_goal)
            if i == len(result)-1:
                text = " ".join(texts[result[i]+1:])
                if text!='':
                    segments.append(text)
                    goals.append(real_goal)
            prev = result[i]+1
    
    dict = {"segment":segments, "goal":goals}
    df = pd.DataFrame(dict)
    df.to_csv('segment_goal_wikiHow.csv', index=False)
    
