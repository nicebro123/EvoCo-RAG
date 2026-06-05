import torch

import time, datetime
def ts(msg: str, t0: float) -> float:
    now = time.time()
    print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S.%f}] {msg} time: {now - t0:.3f} s", flush=True)
    return now


def gen_local(prompt_input_generator,tokenizer,generator,device,train_gen,temperature):
    max_attempts = 3 
    for attempt in range(max_attempts):
        try:
           
            
      
            gen_ = get_gen(prompt_input_generator,tokenizer,generator,device,train_gen,temperature)

            return gen_
            
        except Exception as e:
            print(f"{str(e)}")
            if attempt < max_attempts - 1:
                print("try...")
            else:
                print("fail")
                raise  





def get_gen(prompt_input_generator,tokenizer,generator,device,train_gen,temperature):
    
    if train_gen:
        

        inputs = tokenizer(prompt_input_generator,
                           padding=True, truncation=True,
                           return_tensors="pt",
                           max_length=1024).to(device)

        query_tensors = inputs.input_ids
        prompt_length = query_tensors.size(1)


        with torch.no_grad():
            full_outputs = generator.generate(
                input_ids=query_tensors,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )


        response_tensors = full_outputs[:, prompt_length:]
        response_texts = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )


        return response_texts, query_tensors, response_tensors
    else:


        inputs = tokenizer(prompt_input_generator,
                           padding=True, truncation=True,
                           return_tensors="pt",
                           max_length=2048).to(device)

        query_tensors = inputs.input_ids
        prompt_length = query_tensors.size(1)


        with torch.no_grad():
            full_outputs = generator.generate(
                input_ids=query_tensors,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        response_tensors = full_outputs[:, prompt_length:]
        response_texts = tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        return response_texts, query_tensors, response_tensors



