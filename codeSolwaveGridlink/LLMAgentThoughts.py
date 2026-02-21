from dotenv import load_dotenv
import os
import sys
import glob
import re
import json
import random
import time
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import asyncio
try:
    import sglang as sgl
except ImportError:
    pass
try:
    from vllm import LLM, SamplingParams
    import ray
except ImportError:
    pass

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Missing OPENAI_API_KEY. Put it in your .env as OPENAI_API_KEY=...")

client = OpenAI()

vllm_alias = {
    'mistral': 'mistralai/Mistral-7B-v0.3',
    'mixtral': 'mistralai/Mixtral-8x7B-v0.1',
    'mistral-instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mixtral-instruct': 'mistralai/Mixtral-8x7B-Instruct-v0.1',

    'gemma': 'google/gemma-7b',
    'gemma-2-2b': 'google/gemma-2-2b',
    'gemma-2-2b-it': 'google/gemma-2b-it',
    'gemma-2-27b': 'google/gemma-2-27b',
    'gemma-2-27b-it': 'google/gemma-2-27b-it', # instruction tuned

    'Llama-3-70B': 'meta-llama/Meta-Llama-3-70B',
    'Llama-3-8B': 'meta-llama/Meta-Llama-3-8B',
    'Llama-3.1-405B': 'meta-llama/Meta-Llama-3.1-405B',
    'Llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'Llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B', 
    'Llama-3.1-405B-Instruct': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',

    'Llama-3.3-8B-Instruct': 'meta-llama/Llama-3.3-8B-Instruct',
    'Llama-3.3-70B': 'meta-llama/Llama-3.3-70B',
    'Llama-3.3-8B': 'meta-llama/Llama-3.3-8B',
    'Llama-3.3-70B-Instruct': 'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen3-0.6B': 'Qwen/Qwen3-0.6B',
    'Qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'Qwen3-4B': 'Qwen/Qwen3-4B',
    'Qwen3-8B': 'Qwen/Qwen3-8B',
    'Qwen3-14B': 'Qwen/Qwen3-14B',
    'Qwen3-30B': 'Qwen/Qwen3-30B-A3B',
    'Qwen3-32B': 'Qwen/Qwen3-32B',
    'Qwen3-235B': 'Qwen/Qwen3-235B-A22B',
    'qwen': 'Qwen/Qwen2.5-3B-Instruct',

    'phi-3.5-mini-instruct': 'microsoft/phi-3.5-mini-instruct',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
    'DeepSeek-R1': 'deepseek-ai/DeepSeek-R1',
    'DeepSeek-R1-Distill-Qwen-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'DeepSeek-R1-Distill-Llama-70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',

    'Olmo-3-7B-Think': 'allenai/Olmo-3-7B-Think',
    'Olmo-3-32B-Think': 'allenai/Olmo-3-32B-Think'
}

# global llm and tokenizer store to avoid reloading models
llms = {}
tokenizers = {}

class LLMAgent:
    def __init__(self, model_name: str, config: dict, port: int = None):
        self.model_name = model_name
        self.config = config
        self.port = port

    def _setup_sglang(self, model: str):
        print('using model', model)
        if model in llms:
            print("Info: Model already loaded in llms, skipping setup")
            return
        
        # tokenizer setup
        if model in vllm_alias:
            tokenizers[model] = AutoTokenizer.from_pretrained(vllm_alias[model])
            print('Using tokenizer for', vllm_alias[model])
        else:
            print('Attempting to use tokenizer for', model)
            try:
                tokenizers[model] = AutoTokenizer.from_pretrained(model)
            except Exception:
                print('Info: passing tokenizer setup')

        if model in vllm_alias:
            llms[model] = sgl.Engine(
                model_path=vllm_alias[model], 
                tp_size=self.config['gpus'], 
                random_seed=self.config.get('seed', 0),
                mem_fraction_static=self.config.get('memory_usage', 0.9),
                allow_auto_truncate=self.config.get('allow_auto_truncate', False),
                download_dir=self.config['model_dir']
            )
        elif model and model[0] == '/':
            # load local model
            print("Info: using finetuned model setup")
            llms[model] = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype="auto",
                device_map="auto"
            )
        else:
            try:
                llms[model] = sgl.Engine(
                    model_path=model,
                    tp_size=self.config['gpus'],
                    random_seed=self.config['seed'],
                    mem_fraction_static=self.config.get('memory_usage', 0.9),
                    download_dir=self.config['model_dir']
                )
            except:
                print('Info: Passing vllm setup')


    def _setup_vllm(self, model: str):
        if self.config['gpus'] > 1:
            ray.init(ignore_reinit_error=True, _temp_dir=self.config['tmp_dir']) 

        print("using model ", model)
        if model in llms:
            print("Info: Model already loaded in llms, skipping setup")
            return
        
        # tokenizer setup
        if model in vllm_alias:
            tokenizers[model] = AutoTokenizer.from_pretrained(vllm_alias[model])
            print('Using tokenizer for', vllm_alias[model])
        else:
            print('Attempting to use tokenizer for', model)
            try:
                tokenizers[model] = AutoTokenizer.from_pretrained(model)
            except Exception:
                print('Info: passing tokenizer setup')

        # load vllm
        if model in vllm_alias:
            if "fp8" in self.config.keys() and self.config['fp8']:
                print("Using fp8")
                if vllm_alias[model] in ['meta-llama/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-70B-Instruct']:
                    print("Set up Llama-3.1-70B-Instruct with context length of 8192")
                    llms[model] = LLM(model=vllm_alias[model], tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'], gpu_memory_utilization=self.config.get('memory_usage', 0.65), max_model_len=8192)
                else:
                    llms[model] = LLM(model=vllm_alias[model], tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'], gpu_memory_utilization=self.config.get('memory_usage', 0.85))
            elif vllm_alias[model] in ['meta-llama/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'Qwen/Qwen3-32B']:
                llms[model] = LLM(model=vllm_alias[model], tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'], gpu_memory_utilization=self.config.get('memory_usage', 0.95), max_model_len=12880)
            elif vllm_alias[model] in ['openai/gpt-oss-120b']:
                llms[model] = LLM(model=vllm_alias[model], tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'], gpu_memory_utilization=self.config.get('memory_usage', 0.9), max_model_len=12880)
            else:
                llms[model] = LLM(model=vllm_alias[model], tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'], gpu_memory_utilization=self.config.get('memory_usage', 0.95))
        elif model and model[0] == '/':
            # load local model
            print("Info: using finetuned model setup")
            llms[model] = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            try:
                llms[model] = LLM(model=model, tensor_parallel_size=self.config['gpus'], download_dir=self.config['model_dir'])
            except Exception:
                print('Info: Passing vllm setup')

    def completion_create_port(self, prompt, port, model):
            # print('Prompt:', prompt)
            openai_api_key = "EMPTY"
            openai_api_base = f"http://localhost:{port}/v1" 

            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            # print(prompt)
            completed = False
            if type(prompt) == list:
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]
            while not completed:
                try:
                    chat_response = client.chat.completions.create(model=model, messages=messages)
                    completed = True
                except Exception as e:
                    # # If the API returned a 400 Bad Request, stop retrying and return None
                    # status = None
                    # try:
                    #     status = getattr(e, "status_code", None) or getattr(e, "http_status", None) or getattr(e, "status", None) or getattr(e, "code", None)
                    # except Exception:
                    #     status = None
                    print("Error during OpenAI API call:", e)
                    if "40960" in str(e):
                        print("OpenAI API returned 400 Bad Request via context length; returning None")
                        return None
            ret = {"thoughts": "", "text": "", "response": {"role": "assistant", "content": ""}} # return the output ret at the end and use to calculate cost
            response= chat_response.choices[0].message.content
            if 'Qwen3' in model:
                ret['thoughts'] = response.split("</think>")[0].split("<think>")[-1].strip()
                ret['text'] = response.split("</think>")[-1].strip()
                ret['response'] = {"role": "assistant", "content": ret['text']}
            else:        
                ret['response'] = {"role": "assistant", "content": chat_response.choices[0].message.content}
            return ret

    def completion_create_helper(self, prompt, max_tokens=None, thinking=True):
        model_name = self.model_name
        config = self.config
        
        if "HUMAN" in model_name:
            if isinstance(prompt, list):
                combined_prompt = ""
                for message in prompt:
                    combined_prompt += "\n" + message
                print(combined_prompt) #TODO: agent labels
                # for p in prompt:
                    # print("\n", p)
                if "NO_CONFIRM" not in model_name: # confirm just in case?
                    input("Press Enter when ready to respond...\n")
                response = input("Your response: ")
                return response
            else:
                print("\n", prompt)
                if "NO_CONFIRM" not in model_name:
                    input("Press Enter when ready to respond...\n")
                return input("Your response: ")
        
        if self.port is not None: # local inference server
            return self.completion_create_port(prompt, self.port, vllm_alias.get(model_name, model_name))

        if (model_name in vllm_alias and model_name not in llms) or (model_name[0] == '/' and model_name not in llms):
            if self.config.get('vllm', True):
                self._setup_vllm(model_name)
            else:
                self._setup_sglang(model_name)
                
        tokenizer = tokenizers.get(model_name, None)
        ret = {"thoughts": "", "text": "", "response": {"role": "assistant", "content": ""}} # return the output ret at the end and use to calculate cost
        if isinstance(prompt, list):
            input_key = prompt
        else:
            input_key = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        if model_name in ["o4-mini", "gpt-5-nano", "gpt-5-mini"]:

            if isinstance(prompt, list):
                input_key = prompt
                for msg in input_key: # convert to new formatting
                    if 'content' in msg and isinstance(msg['content'], str) and msg['role'] == 'user':
                        msg['content'] = [
                            {
                                "type": "input_text", 
                                "text": msg['content']
                            }
                        ]
                    elif 'content' in msg and isinstance(msg['content'], str) and msg['role'] == 'assistant':
                        msg['content'] = [
                            {
                                "type": "output_text", 
                                "text": msg['content']
                            }
                        ]
            else:
                input_key = [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "input_text", 
                                "text": prompt
                            }
                        ]
                    }
                ]
            if model_name == "o4-mini":
                reasoning = {"effort": "medium", "summary": "auto"}
            else:
                reasoning = {"effort": "low", "summary": "auto"}

            response = client.responses.create(
                model=model_name,
                reasoning=reasoning,
                input=input_key
            )
            # print(response)
            summary = ""
            for item in response.output:
                if item.type == "reasoning" and len(item.summary) > 0:
                    summary = item.summary[0].text
                elif item.type == "message":
                    # message text is nested under content
                    answer = item.content[0].text
            ret['thoughts'] = summary
            ret['text'] = answer
            ret['response'] = {"role": "assistant", "content": answer}
            if config['verbose']:
                print(ret['response'])
        elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1-nano"]:
            response = client.responses.create(
                model=model_name,
                input=input_key
            )
            summary = ""
            for item in response.output:
                if item.type == "reasoning" and len(item.summary) > 0:
                    summary = item.summary[0].text
                elif item.type == "message":
                    # message text is nested under content
                    answer = item.content[0].text
            ret['thoughts'] = summary
            ret['text'] = answer
            ret['response'] = {"role": "assistant", "content": answer}
            if config['verbose']:
                print(ret['response'])
            
        elif model_name in vllm_alias and model_name in llms:
            if self.config.get('vllm', True):
                sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
            else:
                sampling_params = {"temperature": 0.8, 'top_p': 0.95}
            
            messages = input_key
            if "Qwen3" in model_name:
                prompt_fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            else:
                for msg in input_key: # convert to new formatting
                    if 'content' in msg and isinstance(msg['content'], list) and msg['role'] == 'user':
                        msg['content'] = msg['content'][0]['text']
                    elif 'content' in msg and isinstance(msg['content'], list) and msg['role'] == 'assistant':
                        msg['content'] = msg['content'][0]['text']
                prompt_fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not thinking and "oss" in model_name:
                prompt_fmt += "<|channel|>analysis<|message|><|end|><|start|>assistant"
            # print("prompt_fmt:", prompt_fmt)
            output = llms[model_name].generate([prompt_fmt], sampling_params)

            if thinking:
                if config.get('vllm', True):
                    ret = output[0].outputs[0].text.split("</think>")
                else: # sglang
                    ret = output[0]['text'].split("</think>")
                try:
                    ret = {"thoughts": ret[0], "text": ret[1], "response": {"role": "assistant", "content": ret[1]}}
                except:
                    ret = {"thoughts": "PARSE ERROR", "text": ret[0], "response": {"role": "assistant", "content": ret[0]}} # something went wrong with thoughts
            else:
                if config.get('vllm', True):
                    text = output[0].outputs[0].text
                else:
                    text = output[0]['text']
                ret['text'] = text

            if model_name == 'gpt-oss-120b':
                if not thinking:
                    ret = {"thoughts": "", "text": output[0].outputs[0].text[5:], "response": {"role": "assistant", "content": output[0].outputs[0].text[5:]}} # cut off "final" from beginning of text
                else:
                    text_after_analysis = output[0].outputs[0].text[8:] # cut off initial 'analysis' for thinking in output
                    ret = text_after_analysis.split("assistantfinal") # split between thinking and final response
                    
                    if len(ret) != 2:
                        ret = {"thoughts": "PARSE ERROR", "text": ret[0], "response": {"role": "assistant", "content": ret[0]}} # something went wrong with thoughts
                    else:
                        ret = {"thoughts": ret[0], "text": ret[1], "response": {"role": "assistant", "content": ret[1]}}

        elif model_name == "phi-3.5-mini-instruct": # TODO: refactor
            model_name_hf = "microsoft/phi-3.5-mini-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name_hf, torch_dtype=torch.float16, device_map="auto")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
            ret = {"text": tokenizer.decode(output[0], skip_special_tokens=True)}

        else:
            if "Qwen3" in model_name:
                if max_tokens is None:
                    max_tokens = config.get('max_tokens', 32768)
                messages = input_key
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                inputs = tokenizer([text], return_tensors='pt').to('cuda')
                generated_ids = llms[model_name].generate(**inputs, max_new_tokens=max_tokens)
                output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                    print("value error finding </think> token")
                    print(tokenizer.decode(output_ids, skip_special_tokens=True))
                thinking_content = tokenizer.decode(output_ids[:index-1], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                ret = {"thoughts": thinking_content, "text": content}
            else:
                if max_tokens is None:
                    max_tokens = config.get('max_tokens', 32768)
                if isinstance(prompt, list):
                    inputs = tokenizer.apply_chat_template(input_key, tokenize=True, add_generation_prompt=True)
                else:
                    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    output_ids = llms[model_name].generate(**inputs, max_new_tokens=max_tokens)
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]
                ret = {"thoughts": "", "text": tokenizer.decode(output_ids[0], skip_special_tokens=True), "response": {"role": "assistant", "content": tokenizer.decode(output_ids[0], skip_special_tokens=True)}}

        if config['verbose']:
            print("Response from ", model_name, ": ", ret)
        return ret

    def completion_create(self, prompt, max_tokens=256, keep_trying=True, iter=0, thinking=True):
        try:
            return self.completion_create_helper(prompt, max_tokens=max_tokens, thinking=thinking)
        except (openai.APIError, openai.OpenAIError) as e:
            print(e)
            time.sleep(10)
            if keep_trying and iter < 10:
                return self.completion_create(prompt, max_tokens=max_tokens, keep_trying=keep_trying, iter=iter+1, thinking=thinking)
            else:
                return None

    def __call__(self, prompt, max_tokens=None, keep_trying=False, thinking=True):
        return self.completion_create(prompt, max_tokens=max_tokens, keep_trying=keep_trying, thinking=thinking)

    def __repr__(self):
        return f"LLM: {self.model_name} (Config: {self.config})"






