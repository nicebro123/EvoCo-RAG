import os
import jsonlines

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = []
        for obj in jsonl_f:
            
            pos_non_empty = len(obj['pos_list']) > 0
            neg_non_empty = len(obj['neg_list']) > 0
            
           
            if pos_non_empty and neg_non_empty:
                lst.append(obj)
    print("len(lst):",len(lst),flush=True)
    return lst


def save_jsonlines(data, output_path,mode):

    with jsonlines.open(output_path, mode=mode) as writer:
        for item in data:
            writer.write(item)



def judgeF(ans, answers):
   
    ans = ans.lower()
    answers = [answer.lower() for answer in answers]

    print("\n..begJudge..")
    print("ans: ", ans)
    print("answers: ", answers)
    if ans in answers:
        judge = "yes"
    else:
        judge = "no"
    print("judge: ", judge)
    print("..endJudge..\n")
    return judge




def extract_train_data(data_list):

   
    result_list = []

   
    for data in data_list:

        question = data['question']
       
        has_answer_true = ["title: " + ctx.get('title').strip() + "\ncontext: " + ctx.get('text').strip() for ctx in data['ctxs'] if ctx['hasanswer']]
        has_answer_false = ["title: " + ctx.get('title').strip() + "\ncontext: " + ctx.get('text').strip() for ctx in data['ctxs'] if not ctx['hasanswer']]

        if len(has_answer_true) != 0 and len(has_answer_false) != 0:
          
            result = {
                'query': question,
                'pos_list': has_answer_true,
                'neg_list': has_answer_false
            }
          
            result_list.append(result)
    return result_list





def extract_test_data(data_list):
    result_list = []
   
    for data in data_list:
      
        question = data.get('question')
        answers = data.get('answers', [])
        context = ["title: " + ctx.get('title').strip() + "\ncontext: " + ctx.get('text').strip() for ctx in data.get('ctxs', [])]
        ground_truth =[ctx.get('ground_truth') for ctx in data.get('ctxs', [])]
        naive_rag =[ctx.get('naive_rag') for ctx in data.get('ctxs', [])]

      
        result = {
        "question": question,
        "answers": answers,
        "context": context,
        "ground_truth": ground_truth,
        "naive_rag": naive_rag
        }

        result_list.append(result)

    return result_list



def data_process():
 
    data_list = load_jsonlines("popqa_train.jsonl")
    # print(len(data_list))
    extract_data = extract_train_data(data_list)
    save_jsonlines(extract_data, "popqa_train_process.jsonl")

    extract_data_list = load_jsonlines("popqa_train_process.jsonl")  
    print(len(extract_data_list))

    # test
    data_list = load_jsonlines("popqa_test.jsonl")
    extract_data = extract_test_data(data_list)  
    save_jsonlines(extract_data, "popqa_test_process.jsonl")

    extract_data_list = load_jsonlines("popqa_test_process.jsonl")  
    print(extract_data_list[0])



import io
import json
import re, json, string
import numpy as np
from tqdm import tqdm


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    if question.startswith("."):
        question = question.lstrip(". ")

    return question[0].lower() + question[1:]


def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['answers'], item["rationale"]))

        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def exact_presence(answers, context):
    """Verify if any of the answers is present in the given context."""

    answers = [normalize_answer(ans) for ans in answers]
    context = normalize_answer(context)

    for ans in answers:
        if ans in context:
            return True

    return False





def get_metrics(data, e,dataty, is_asqa=False):
    idx = 0
    num_accurate = 0
    print('\n\n\nEvaluating results...\n')
    if is_asqa:
        rationale_str_em, _ = compute_str_em(data)
    else:
        for d in tqdm(data):
            idx += 1
            # print("测试：\n")
            # print("answers：\n",d['answers'])
            # print("rationale：\n", d['rationale'])
            is_accurate = exact_presence(d['answers'], d['rationale'])
            print("is_accurate: ",is_accurate)
            num_accurate += 1 if is_accurate else 0

    if is_asqa:
        print(f"Rationale EM: {rationale_str_em:.1f}%")
        eval_result = {"EM": rationale_str_em, "num_examples": idx}
    else:
        print("num_accurate :",num_accurate)
        print("idx :",idx)
        accuracy = num_accurate / idx * 100
        print(f"Accuracy for Epoch {e}: {accuracy:.1f}%")
        eval_result = {"dataty": dataty, "Epoch": e, "accuracy": accuracy, "num_examples": idx}
    

    return eval_result