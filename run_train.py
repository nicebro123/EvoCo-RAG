import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from tqdm import tqdm
import random
from utils import *
from run_test import run_test
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig
import json
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from trl import GRPOConfig,GRPOTrainer
import time, datetime
import logging, json
import math
import warnings
warnings.filterwarnings("ignore")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.join(os.path.dirname(REPO_DIR), "rag_assets")
LEGACY_ROOT = os.path.join(ASSET_ROOT, "legacy")
LEGACY_DATA_ROOT = os.path.join(LEGACY_ROOT, "data")
LEGACY_ADAPTER_ROOT = os.path.join(LEGACY_ROOT, "adapters")
LEGACY_OUTPUT_ROOT = os.path.join(LEGACY_ROOT, "outputs")
BASE_RERANKER_PATH = os.path.join(ASSET_ROOT, "base_models", "reranker", "bge-reranker-v2-m3")
BASE_GENERATOR_PATH = os.path.join(ASSET_ROOT, "base_models", "generator", "Mistral-Nemo-Instruct-2407")


def ts(msg: str, t0: float) -> float:
    now = time.time()
    print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S.%f}] {msg} 耗时: {now - t0:.3f} s", flush=True)
    return now


def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    lst = []
    for item in data:
        lst.append(item)
    print("len(lst):", len(lst), flush=True)
    return lst


def load_jsonlines(file):
    lst = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            lst.append(item)
    print("len(lst):", len(lst), flush=True)
    return lst




def compute_reward(que, 
                   res, 
                   gold_list, 
                   w_sub = 1.0,
                   w_smi = 0.0,
                   w_pro = 0.0
    ):
    
    """数据处理"""
    res = normalize_answer(res)
    res_process = res
    if "final answer" in res:
            res_process = res.split("final answer", 1)[1].strip()  

    gold_list = [normalize_answer(gold) for gold in gold_list]
    

    best_reward = -100
    best_reward_str = ''

    for gold in gold_list:
        

        if gold in res_process:
            substring_reward = 1.0
        else:
            substring_reward = 0
        


        combine_r = w_sub * substring_reward 
        combine_r_str =f"best_combine_r: {combine_r} = {w_sub} * {substring_reward}"
        
        if combine_r > best_reward:
            best_reward = combine_r
            best_reward_str = combine_r_str
    print(best_reward_str)
    return best_reward






def myReward(completions, **kwargs):

    n_per_train = kwargs.get('n_per_train', [])[0]
    num_generations = kwargs.get('num_generations', [])[0]
    epoch = kwargs.get('epoch', [])[0]
    dataTy = kwargs.get('dataTy', [])[0]
    ver = kwargs.get('ver', [])[0]
    extract_data_list_path = kwargs.get('extract_data_list_path', [])[0]

    print("len(completions):",len(completions))
    print(f"epoch: {epoch}")
    print(f"dataTy: {dataTy}")
    print(f"ver: {ver}")
    


    question_list = kwargs.get('question', [])
    top1_doc_list = kwargs.get('top1_doc', [])
    top1_index_list = kwargs.get('top1_index', [])
    answers_list = kwargs.get('answers', [])
    pad_data_index_list = kwargs.get('pad_data_index', [])

    reward_list = []
    ver_data_list = []
    for i in range(n_per_train):
        index = i*num_generations   

        question = question_list[index]
        top1_doc = top1_doc_list[index]
        top1_index = top1_index_list[index]
        answers = answers_list[index]
        pad_data_index = pad_data_index_list[index]
        
        

        if epoch == 0:
            extract_data_list = load_json(extract_data_list_path)
        else:
            extract_data_list = load_jsonlines(extract_data_list_path)

       
        pad_data = extract_data_list[pad_data_index]
       

    
        Judgeflag = 0  
        for j, completion in enumerate(completions[index : index + num_generations]):
            c = completion[0]['content']
            is_accurate = exact_presence(answers, c) 
            if is_accurate==True:
                Judgeflag = 1
            print(f"\n\nCompletion[{j}] for batch_data[{i}]\n",c,"\nJudge: ",is_accurate)

            reward = compute_reward(question,c,answers)  
            reward_list.append(reward)
        

        new_labels = pad_data["labels"] 
        if Judgeflag:
            new_labels[top1_index].append("1")
        else:
            new_labels[top1_index].append("0")
            
        ver_data = {
                "question":pad_data["question"],
                "answers":pad_data["answers"],
                "context":pad_data["context"],
                "labels":new_labels
            }

    
        ver_data_list.append(ver_data)
    
    next_epoch_path = os.path.join(LEGACY_DATA_ROOT, f"data_v{ver}", dataTy, f"epoch_{epoch+1}.jsonl")
    os.makedirs(os.path.dirname(next_epoch_path), exist_ok=True)
    with open(next_epoch_path, "a", encoding="utf-8") as f:
                for ver_data in ver_data_list:
                    f.write(json.dumps(ver_data, ensure_ascii=False) + "\n")
        
    print("\n\nreward_list:\n",reward_list,"\n\n\n")   
    return reward_list



def load_ranker(ranker_path, use_lora=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = AutoModelForSequenceClassification.from_pretrained(ranker_path,).to(device)
    #print(ranker)
    
    if use_lora:
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["query", "key", "value", "attention.output.dense", 
                          "intermediate.dense", "output.dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        ranker = get_peft_model(ranker, lora_config)
        for name, param in ranker.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        ranker.print_trainable_parameters()
    
    ranker_tokenizer = AutoTokenizer.from_pretrained(ranker_path)
    return ranker, ranker_tokenizer, device




class RankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, batch_scores_list, batch_pos_lens):
        batch_loss = 0.0
        valid_count = 0
        for i in range(len(batch_scores_list)):
            pos_len = batch_pos_lens[i]
            neg_len = len(batch_scores_list[i]) - pos_len
            if pos_len == 0 or neg_len == 0:
                continue
            pos_scores = batch_scores_list[i][0:pos_len]
            neg_scores = batch_scores_list[i][pos_len:]
            for pos_score in pos_scores:
                batch_loss = batch_loss + torch.sum(
                    F.relu(neg_scores - pos_score + self.margin)
                )
            valid_count += 1
        if valid_count > 0:
            batch_loss = batch_loss / valid_count
        return batch_loss


def calculate_positive_probability(label):
    count_ones = label.count("1")
    total = len(label)
    p = count_ones / total
    mapped_p = 0.1 + p * 0.8
    return mapped_p


def is_positive_document(label, seed=42):
    p = calculate_positive_probability(label)
    r = random.random()
    return r < p




# ============================================
# reranker Train
# ============================================
def reranker_training(device, extract_data_list, ranker, tokenizer_ranker, optimizer_ranker, epoch, dataTy, batch_size, ver):


    all_results = []

    for start_idx in tqdm(range(0, len(extract_data_list), batch_size), desc="Batch Progress", leave=False):

        batch_data = extract_data_list[start_idx:start_idx + batch_size] 
        

        t0 = time.time()   
        

        batch_queries = [d["question"] for d in batch_data]
        batch_answers = [d["answers"] for d in batch_data]
        

        batch_pos_lists = []  
        batch_neg_lists = [] 
        
        for d in batch_data:
            pos_list = []
            neg_list = []
            

            for idx, (context, label) in enumerate(zip(d["context"], d["labels"])):
                doc_info = {"idx": idx, "doc": context}      

                if is_positive_document(label):
                    pos_list.append(doc_info)
                else:
                    neg_list.append(doc_info)
            
            batch_pos_lists.append(pos_list)
            batch_neg_lists.append(neg_list)
    
        

        ranker.train()  
        all_pair_list = [] 
        split_indices = [0] 
        total_count = 0    
        batch_top1_docs = [] 
        batch_docs = []    
        

        for i in range(len(batch_queries)):
            pos_docs = batch_pos_lists[i]
            neg_docs = batch_neg_lists[i]

            batch_docs.extend(pos_docs + neg_docs)

            cur_pairs = [(batch_queries[i], item["doc"]) for item in pos_docs + neg_docs]
            
            all_pair_list.extend(cur_pairs)
            total_count += len(cur_pairs) 
            split_indices.append(total_count)
            

        inputs = tokenizer_ranker(
            all_pair_list, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512
        ).to(device)
        

        all_scores = ranker(**inputs, return_dict=True).logits.view(-1)
        

        scores_list_batch = []  
        top1_index_batch = []  
        
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            
            index_max = torch.argmax(all_scores[start:end])
            

            context = batch_docs[start:end]

            top1_doc_dict = context[index_max]
            
            top1_doc = top1_doc_dict["doc"]
            top1_idx = top1_doc_dict["idx"]
            
            top1_index_batch.append(top1_idx)
            batch_top1_docs.append(top1_doc)
            scores_list_batch.append(all_scores[start:end])

        
            
        ranking_loss = RankingLoss()
        batch_pos_lens = []  
        flagT = 0          
        

        for i in range(len(batch_pos_lists)):
            pos_len = len(batch_pos_lists[i])
            neg_len = len(batch_neg_lists[i])

            if pos_len == 0 or neg_len == 0:
                flagT += 1
            batch_pos_lens.append(pos_len)
        

        t1 = ts("\nRanker \n", t0)
    

        if flagT != batch_size:
            batch_loss = ranking_loss(scores_list_batch, batch_pos_lens)
            print(f"\nReranker batch_loss: {batch_loss.item():.4f}", flush=True)
            

            optimizer_ranker.zero_grad()  
            batch_loss.backward()        
            optimizer_ranker.step()     
        else:
            print("\nskip reranker!", flush=True)
    

        t2 = ts("Ranker Optim\n", t1)



        for i in range(len(batch_queries)):
            result_dict = {
                "question": batch_queries[i],
                "top1_doc": batch_top1_docs[i],
                "top1_index": int(top1_index_batch[i]),  
                "answers":batch_answers[i],
                "pad_data_index":start_idx + i, 
                "epoch":epoch,
                "dataTy":dataTy,
                "ver":ver
            }
            all_results.append(result_dict)
            
   
    save_path = os.path.join(LEGACY_DATA_ROOT, f"data_v{ver}", dataTy, f"epoch_{epoch}_reranker_results.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n{save_path}")








if __name__ == "__main__":
    

    ver = 33
    batch_size = 4
    epochNum = 10
    methodTy = f'Stage3_V{ver}_GRPO'
    dataTy = "Pop"
    modelTy = "Base"
    num_generations = 2
    n_per_train = 16
    use_4bit = False  # 默认 bf16（H20 显存充足，避免量化精度损失）；置 True 走 nf4 4bit



    print("\nRanker init")
    ranker_path = BASE_RERANKER_PATH
    ranker, tokenizer_ranker, device = load_ranker(ranker_path)

    ranker.print_trainable_parameters()
    optimizer_ranker = Adam(filter(lambda p: p.requires_grad, ranker.parameters()), lr=5e-5)


    print("\nGenerator init")
    base_model = BASE_GENERATOR_PATH
    tokenizer_llm = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id
    tokenizer_llm.padding_side = 'left'
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"\n✓ LLM device_map: {llm_model.hf_device_map}\n")
        llm_model = prepare_model_for_kbit_training(llm_model, use_gradient_checkpointing=False)
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"\n✓ LLM device_map: {llm_model.hf_device_map}\n")

    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm_model = get_peft_model(llm_model, lora_config)

    llm_model.print_trainable_parameters()


  
    #GRPOConfig
    training_args = GRPOConfig(
        output_dir=os.path.join(LEGACY_OUTPUT_ROOT, f"grpo_output{ver}"),
        save_strategy="no", 

        num_train_epochs=1,         
        num_generations=num_generations,            
        per_device_train_batch_size=num_generations * n_per_train,
        learning_rate=1e-5,          


        max_prompt_length = 1024,
        max_completion_length = 512,
        shuffle_dataset =False,
        report_to="none",
        
        logging_steps=1,
        logging_first_step=True,
        logging_strategy="steps",
        bf16=True,
        fp16=False,
    )
    print("GRPOTrainerinit")
    grpoTrainer = GRPOTrainer(
            model=llm_model,          
            args=training_args,       
            train_dataset=None,       
            reward_funcs=myReward,   
        )
     


    print("\nload ")
    extract_data_list_path = os.path.join(ASSET_ROOT, f"data_v{ver}", dataTy, "train_labels_list.json")
    extract_data_list = load_json(extract_data_list_path)
    target_len = (len(extract_data_list) // n_per_train) * n_per_train
    extract_data_list = extract_data_list[:target_len]
    print("len(clip_lst): ", len(extract_data_list), flush=True)

    #extract_data_list = extract_data_list[:n_per_train]  



 
    best_acc = -1
    
    for epoch in tqdm(range(epochNum), desc="Epoch Progress"):
        if epoch != 0:
            extract_data_list_path = os.path.join(LEGACY_DATA_ROOT, f"data_v{ver}", dataTy, f"epoch_{epoch}.jsonl")
            extract_data_list = load_jsonlines(extract_data_list_path)

        print(f"\n\nbegin reranker_training epoch {epoch}...")
        reranker_training(
            device=device,
            extract_data_list=extract_data_list,
            ranker=ranker,
            tokenizer_ranker=tokenizer_ranker,
            optimizer_ranker=optimizer_ranker,
            epoch=epoch,
            dataTy=dataTy,
            batch_size=batch_size,
            ver=ver,
        )
        print(f"\n\nfinished reranker_training epoch {epoch}!")

        
        print(f"\n\nbegin generator_training epoch {epoch}...")
        reranker_choose_path = os.path.join(LEGACY_DATA_ROOT, f"data_v{ver}", dataTy, f"epoch_{epoch}_reranker_results.json")
        reranker_choose_dataset = load_dataset("json", 
                                               data_files=reranker_choose_path, 
                                               split="train")

        
        SYSTEM_PROMPT = '''
You are tasked with answering the given question by analyzing a set of documents. Please follow this STRICT TWO-STEP PROCESS:

    ---

    ### STEP 1: Document Analysis  
    For each document:
    - First, extract potentially relevant information from the original document. This includes facts, names, dates, or statements that may relate to the question, even if the connection is not immediately obvious.  
    - Then, explain the reason for the information extraction in your previous step based on the question. (e.g., how the document addresses the question’s focus).

    ### STEP 2: Final Answer   
    - Summarize your answer to the question based on the analysis above.  
    - If none of the documents are helpful or relevant, answer based on your own general knowledge. In that case, clearly state that you are doing so.

    ---

    Use the following format for your response:

    ### Step 1: Document Analysis  
    Document 1:  
    - Extraction: ...  
    - Explanation: ...  

    Document 2:  
    - Extraction: ...  
    - Explanation: ...  
    ...

    ### Step 2: Final Answer  
    Well-supported answer, based on the relevant documents. If no relevant documents, answer based on general knowledge and say so explicitly.

    ---

EXAMPLE:

    Question: Who is the author of The Mahdi?

    ### Step 1: Document Analysis  
    Document 1:  
    - Extraction:  'Mahdi' is a thriller novel by Philip Nicholson written in 1981 under the identity of a. J. Quinnell ...
    - Explanation: This document directly states that the author of 'Mahdi' is Philip Nicholson. Upon re-examining the question "Who is the author of The Mahdi?", the document mentions the corresponding book title and provides the author information requested.

    Document 2:  
    - Extraction: 'The Mahdi' was published by Philip Nicholson in 1981 under the pen name A.J. Quinnell, establishing his presence in the thriller genre with a novel known for its gripping plot and enduring popularity...  
    - Explanation: This document directly states that 'The Mahdi' was published by Philip Nicholson. Upon re-examining the question "Who is the author of The Mahdi?", the document mentions the corresponding book title and provides the author information requested.

    ### Step 2: Final Answer  
    Based on Documents 1 and 2, The answer to question 'Who is the author of The Mahdi?' is A.J. Quinnell, a pseudonym of Philip Nicholson.

    ###
'''     
        
        reranker_choose_dataset = reranker_choose_dataset.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': 
            f'''
Now it is your turn to analyze the following documents and answer the given question by following the two-step process.

Document 1 {x['top1_doc']}

Based on your knowledge and the provided information, answer the question:
{x['question']}
            '''
        }
            ],
            "top1_doc": x['top1_doc'],
            "top1_index": x['top1_index'],
            "answers": x['answers'],
            "extract_data_list_path":extract_data_list_path,
            "n_per_train":n_per_train,
            "num_generations":num_generations

        })

        print("\nGRPOTrainer set Dataset")
        grpoTrainer.train_dataset = reranker_choose_dataset

        print("\ntrain")
        grpoTrainer.train()
        print(f"\n\nfinished generator_training epoch {epoch}!")

        

        print(f"\n\nsave...")
        ranker.save_pretrained(os.path.join(LEGACY_ADAPTER_ROOT, f"reranker{methodTy}"))
        llm_model.save_pretrained(os.path.join(LEGACY_ADAPTER_ROOT, f"generator{methodTy}"))
        tokenizer_llm.save_pretrained(os.path.join(LEGACY_ADAPTER_ROOT, f"generator{methodTy}"))
        
        # test
        ACC = run_test(e=epoch, dataTy=dataTy, modelTy=modelTy, stage=3, methodTy=methodTy, use_4bit=use_4bit)
        if ACC > best_acc:
            best_acc = ACC
            ranker.save_pretrained(os.path.join(LEGACY_ADAPTER_ROOT, f"reranker{methodTy}_best"))
            llm_model.save_pretrained(os.path.join(LEGACY_ADAPTER_ROOT, f"generator{methodTy}_best"))
            print(f"\nepoch {epoch}touch best {ACC:.4f}\n\n", flush=True)
    
    print("\ntrain finished！", flush=True)
