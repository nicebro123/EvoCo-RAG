import torch
from utils import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from llm_local_prompt import gen_local
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel
import io
import json
import os
from peft import  get_peft_model, LoraConfig

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = os.path.join(os.path.dirname(REPO_DIR), "rag_assets")
LEGACY_ADAPTER_ROOT = os.path.join(ASSET_ROOT, "legacy", "adapters")
BASE_RERANKER_PATH = os.path.join(ASSET_ROOT, "base_models", "reranker", "bge-reranker-v2-m3")
FT_RERANKER_PATH = os.path.join(ASSET_ROOT, "base_models", "reranker", "bge-reranker-v2-m3-ft")
BASE_GENERATOR_PATH = os.path.join(ASSET_ROOT, "base_models", "generator", "Mistral-Nemo-Instruct-2407")


def load_ranker(ranker_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = AutoModelForSequenceClassification.from_pretrained(ranker_path,).to(device)
    ranker_tokenizer = AutoTokenizer.from_pretrained(ranker_path)
    return ranker, ranker_tokenizer, device





def create_test_batches(test_data_list, batch_size):
    batches = []
    for i in range(0, len(test_data_list), batch_size):
        batch = test_data_list[i:i+batch_size] 
        batches.append(batch)
    return batches



def do_test(ranker, ranker_tokenizer, tokenizer_llm, llm_model, device, train_gen, test_batches, e,dataty):
    dic_list = []
    for batch_idx, batch in enumerate(tqdm(test_batches, desc=f"\nTest Batch Progress")):
        print(f"\nProcessing Batch {batch_idx+1}/{len(test_batches)}", flush=True)
        

        batch_pair_list = [] 
        batch_meta = []      
        for data in batch:
            question = data["question"]
            answers = data["answers"]
            context = data["context"]  
            num_docs = len(context)   
            

            sample_pairs = [(question, doc) for doc in context]
            batch_pair_list.extend(sample_pairs)
            

            batch_meta.append( (question, answers, num_docs, context) )


        with torch.no_grad(): 
            inputs = ranker_tokenizer(
                batch_pair_list,
                padding=True,   
                truncation=True, 
                return_tensors='pt',
                max_length=512
            ).to(device)  
            
            scores = ranker(**inputs, return_dict=True).logits.view(-1).float().cpu() 


        batch_scores = [] 
        score_ptr = 0     
        for (_, _, num_docs, _) in batch_meta:
            sample_scores = scores[score_ptr : score_ptr + num_docs]
            batch_scores.append(sample_scores)
            score_ptr += num_docs 


        batch_prompts = [] 
        batch_answers = [] 
        for idx, (question, answers, _, context) in enumerate(batch_meta):

            sample_scores = batch_scores[idx]
            top_n_indices = torch.argsort(sample_scores, descending=True)[:3].tolist()  # Top-5
            passages = [context[i] for i in top_n_indices]
            

            #prefix
            prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
            prefix += '''
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

            prefix += "\n\nNow it is your turn to analyze the following documents and answer the given question by following the two-step process.\n\n"
           

            docs_text = "\n\n".join([f"Document {doc_idx+1} {ctx}" for doc_idx, ctx in enumerate(passages)])
            doc_prompt = f"{docs_text}\n\n"


            query_prompt = f"Based on your knowledge and the provided information, answer the question:\n{question}"

            target_prefix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            formatted_prompt = prefix + doc_prompt + query_prompt + target_prefix
            

            batch_prompts.append(formatted_prompt)
            batch_answers.append(answers)


        print(f"\nBatch {batch_idx+1}: Generating answers for {len(batch_prompts)} samples", flush=True)

        responses, query_tensors, response_tensors  = gen_local(
            prompt_input_generator=batch_prompts,  
            tokenizer=tokenizer_llm,
            generator=llm_model,
            device=device,
            train_gen=False,
            temperature=0.7
        )


        for ans, resp, prompt in zip(batch_answers, responses, batch_prompts):
            print(f"\n\nBatch {batch_idx+1} - Sample Result:", flush=True)
            print(f"Prompt: {prompt}", flush=True)  
            print(f"Response: {resp}", flush=True)  
            dic = {
                "answers": ans,
                "rationale": resp
            }
            dic_list.append(dic)

    eval_result = get_metrics(dic_list, e, dataty, is_asqa=False)
    return eval_result





def load_json_for_test(file):
    with open(file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    result_list =[]
    for data in test_data:
        question = data['question']
        answers = data['answers']
        context = ["title: " + ctx.get('title').strip() + "\ncontext: " + ctx.get('text').strip() for ctx in data['ctxs']]

        result = {
            'question': question,
            'answers': answers,
            'context': context
        }
        result_list.append(result)
    return  result_list
    




def run_test(e, dataTy,modelTy, stage, methodTy,batch_size=16, use_4bit=False):
    print(f"\n{modelTy}{dataTy}{stage}...",flush=True)

    #reranker
    if modelTy=="FT":
        RERANKER_PATH = FT_RERANKER_PATH
    else:
        RERANKER_PATH = BASE_RERANKER_PATH

    ranker, ranker_tokenizer, device = load_ranker(RERANKER_PATH)


    reranker_dir = os.path.join(LEGACY_ADAPTER_ROOT, f"reranker{methodTy}")
    ranker = PeftModel.from_pretrained(ranker, reranker_dir)
    ranker.eval() 


    #generator
    base_model = BASE_GENERATOR_PATH

    tokenizer_llm = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id
    tokenizer_llm.padding_side = 'left'

    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    generator_dir = os.path.join(LEGACY_ADAPTER_ROOT, f"generator{methodTy}")
    print(f"\n generator Lora: {generator_dir}\n",flush=True)
    llm_model = PeftModel.from_pretrained(llm_model, generator_dir)

    ###################################################################

    llm_model.eval()
    llm_model.gradient_checkpointing_disable()  
    ###################################################################



    test_data_list = load_json_for_test(os.path.join(ASSET_ROOT, "data", dataTy, "test.json"))
    #调试使用
  



    test_batches = create_test_batches(test_data_list, batch_size=batch_size)


    eval_result = do_test(
        ranker, 
        ranker_tokenizer,
        tokenizer_llm,
        llm_model=llm_model,
        train_gen=False,
        device=device,
        test_batches=test_batches,  
        e=e,
        dataty=dataTy
    )
    return eval_result["accuracy"]

