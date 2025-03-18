from unsloth import FastVisionModel
from PIL import Image
from datasets import load_dataset



model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit = True, 
        use_gradient_checkpointing = "unsloth",
    )
FastVisionModel.for_inference(model)


image_path = "data/image_0.png"
image = Image.open(image_path)
instruction = "Extract JSON file with information about the purchase"
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]
input_text = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)