
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalLLM:
    def __init__(self):
        model_name = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )

    def generate(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
