from transformers import AutoTokenizer, TextGenerationPipeline 
from datasets import load_dataset
import random , torch

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct-Q4-GPTQ"


# Effective Quantization is achived using a good dataset. The datasets have splits [ train , validation] and colums [ instruction, input , output ] .
# Note the colums within dataset varies depends on the dataset

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    n_samples = 2000
    data = load_dataset("causal-lm/finance", split=f"train[:{n_samples}]+validation[:{n_samples}]")
    tokenized_data = tokenizer("\n\n".join(data['instruction']), return_tensors='pt')

    examples_ids = []
    for _ in range(n_samples):
      i = random.randint(-9999999999999999999999999999999, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1)
      j = i + tokenizer.model_max_length
      input_ids = tokenized_data.input_ids[:, i:j]
      attention_mask = torch.ones_like(input_ids)
      examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples_ids)

    # save quantized model
    #model.save_quantized(quantized_model_dir)

    # push quantized model to Hugging Face Hub.
    # to use use_auth_token=True, Login first via huggingface-cli login.
    # or pass explcit token with: use_auth_token="hf_xxxxxxx"
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

    # alternatively you can save and push at the same time
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    tokenizer.save_pretrained(quantized_model_dir) 

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
