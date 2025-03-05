from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
print("已完成-------------在所有函数之前使用 PatchFastRL 来修复 GRPO 和其他强化学习算法！-------------")
import re
import time
import openai
import sqlparse
from openai import OpenAI
from datasets import load_dataset, Dataset

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 10240 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./Qwen2.5-7B-SFT-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
print(type(model))
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", 
        "k_proj",
        "v_proj", 
        "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
print(type(model))
print("已完成-------------加载模型和设置参数-------------")

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 5000,
    max_completion_length = 500,
    num_train_epochs = 1, # Set to 1 for a full training run
    #max_steps = -1,
    save_steps = 500,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs_grpo",
)

print("----------------训练参数配置完成=================")

# Load and prep dataset
SYSTEM_PROMPT = """
#Task:Convert natural language into SQL queries based on user's questions,relevant evidence and Database_Schema that can retrieve relevant data from the database.
#Rule:Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
```sql
...
```
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# uncomment middle messages for 1-shot prompting
def get_questions(split = "train") -> Dataset:
    data = load_dataset('./data')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer']
    }) # type: ignore
    return data # type: ignore
    
dataset = get_questions()
print(dataset)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions

client = OpenAI(
    api_key = "",
    base_url = "",
    timeout = 5
)

def generate_prompt(sql1: str, gold_sql: str) -> str:
    # 定义 Prompt 模板
    prompt_template = """
    ## 任务说明
    作为数据库专家，分析predict_SQL和Gold_SQL的语义相似性，按照分析要求，完全相似（判断predict_SQL和Gold_SQL查询结果应该一致）给2分，否则0分。只打分，不做说明！  
    ## 输入数据
    【predict_SQL】:
    {sql1}   
    【Gold_SQL】:
    {gold_sql} 
    ## 分析要求
    按以下维度进行对比：
    1. **查询目标分析**
       - 比较SELECT子句中的目标列（忽略列顺序和别名，但检查内容）
       - 验证是否返回相同的数据类型   
    2. **数据来源分析**
       - 对比FROM子句和JOIN逻辑（列可以）
       - 检查表别名使用是否影响语义
       - 验证连接条件是否逻辑等价   
    3. **过滤条件分析**
       - 比较WHERE条件中的逻辑表达式
       - 检查隐式条件（如JOIN中的ON条件）
       - 验证NULL处理是否一致  
    4. **聚合与分组**
       - 对比GROUP BY字段和聚合函数
       - 检查HAVING条件逻辑  
    5. **排序与限制**
       - 比较ORDER BY字段和排序方向
       - 验证LIMIT/OFFSET值是否匹配  
    6. **子查询与CTE**
       - 检查嵌套查询的逻辑等价性
       - 比较CTE与内联子查询的对应关系
    ## 输出格式
    <score>
    ...
    </score>
    """
    # 使用 str.format() 填充模板
    return prompt_template.format(sql1=sql1, gold_sql=gold_sql)

def call_llm(prompt: str, model: str = "ep-20250216123327-pcpt7", 
            max_retries: int = 3, retry_delay: float = 1.5) -> str:
    """
    增强版LLM调用函数，包含：
    - 指数退避重试机制
    - 智能超时检测
    - 错误类型识别
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
                timeout=8  # 单次请求超时时间
            )
            return response.choices[0].message.content.strip()
        
        except openai.APITimeoutError as e:
            print(f"Timeout occurred (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            retry_count += 1
            time.sleep(retry_delay * (2 ** retry_count))  # 指数退避
            
        except openai.APIError as e:
            error_code = e.response.status_code if e.response else "Unknown"
            if error_code in [429, 500, 503]:  # 可恢复的错误代码
                print(f"Recoverable error {error_code} (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(retry_delay)
            else:
                return f"<score>ERROR: {str(e)}</score>"
            
        except Exception as e:
            return f"<score>FATAL: {str(e)}</score>"

    return f"<score>ERROR: Max retries ({max_retries}) exceeded</score>"
    
def extract_score(output: str) -> float:
    """
    从包含<score>标签的文本中提取数值，支持以下格式：
    - 标准格式：<score>1.5</score>
    - 带换行符：<score>\n 2 \n</score>
    - 错误格式：<score>ERROR</score>
    
    返回：
    - 成功时返回浮点数
    - 无法解析时返回-1.0
    """
    # 正则表达式模式（支持小数、整数、科学计数法）
    pattern = r"<score>\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)\s*</score>"
    
    match = re.search(pattern, output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0
    else:
        # 处理特殊错误格式
        if "<score>" in output and "</score>" in output:
            return 0
        # 完全无标签的情况
        return 0
        
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    responses = [completion[0]['content'] for completion in completions]
    question = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(response) for response in responses]
    print('-'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for r, a in zip(extracted_responses, answer):
        prompt = generate_prompt(r, a)
        reply = call_llm(prompt)
        reward = extract_score(reply)  
        rewards.append(reward)
    return rewards
    
# def is_sql_statement(answer: str) -> bool:
#     """
#     检查 answer 是否是以 ```sql 开头和 ``` 结尾的 SQL 语句。
#     """
#     # 检查是否以 ```sql 开头，并以 ``` 结尾
#     return answer.strip().startswith("```sql") and answer.strip().endswith("```")
# def sql_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if is_sql_statement(r) else 0.0 for r in extracted_responses]


def extract_sql_answer(text):
    """从生成的文本中提取 SQL 查询"""
    # 匹配 <answer> 标签内的 SQL 内容，考虑任意空白字符
    pattern = r"<answer>\s*```sql\s*(.*?)\s*```[\s\S]*?</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # 使用 sqlparse 格式化 SQL 查询，忽略换行符和多余空格
        formatted_sql = sqlparse.format(match.group(1), strip_comments=True, reindent=False, keyword_case='upper').replace('\n', ' ').strip() 
        return formatted_sql  
    return ""
    

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\s*([\s\S]+?)\s*</reasoning>\s*<answer>\s*```sql\s*([\s\S]+?)\s*```\s*</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    print("strict_format提取的responses", responses[0])
    print("打印repr(responses[0])：",repr(responses[0]))
    print("strict_format正则匹配结果:", matches[0])
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>\s*([\s\S]+?)\s*</reasoning>\s*<answer>\s*([\s\S]+?)\s*</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
    
print("-------------------------数据准备完成--------------------")
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func
    ],
    args = training_args,
    train_dataset = dataset
)
print("---------------------开始训练-------------------")
trainer.train()
model.save_lora("grpo_saved_lora")
model.save_pretrained_merged("Qwen2.5-7B-SFT-GRPO", tokenizer, save_method = "merged_16bit",)
