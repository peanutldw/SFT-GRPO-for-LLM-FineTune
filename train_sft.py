from unsloth import FastLanguageModel
import torch

max_seq_length = 10240 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj",
                      #"k_proj",
                      "v_proj",
                      #"o_proj",
                      #"gate_proj", "up_proj", "down_proj"
                      ],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

PROMPT_TEMPLATE = """###Task:Convert natural language into SQL queries based on user's questions,relevant evidence and Database_Schema that can retrieve relevant data from the database. 
{question}

###Response:
{answer}"""

EOS_TOKEN = tokenizer.eos_token

def format_qa(examples):
    texts = []
    inputs = examples["question"]
    outputs = examples["answer"]
    # 遍历每个样本的question/answer对
    for input, output in zip(inputs, outputs):
        # 将模板中的占位符替换为实际内容
        text = PROMPT_TEMPLATE.format(question=input, answer=output) + EOS_TOKEN
        texts.append(text)
    return { "text": texts, }
pass

# 加载数据集（假设你的数据文件名为 data.json）
from datasets import load_dataset
dataset = load_dataset("data", split = "train")

# 应用格式化函数
dataset = dataset.map(format_qa, batched=True)

# 查看结果样例
print(dataset[0]["text"])

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit_forced",)