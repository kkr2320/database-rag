import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# Load pre-trained LLMA3 model and tokenizer
model_name_or_path = "./meta-llama/Meta-Llama-3-8B-Instruct-awq"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Set device for the model (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a lock to ensure thread-safe access to the model
model_lock = threading.Lock()

# Define an API endpoint to handle user requests
@app.route('/generate', methods=['POST'])
def generate_response():
    # Get the input text from the request
    input_text = request.get_json()['input']

    # Construct the prompt template using the input text and a predefined template
    prompt_template = f"Q: {input_text} A: "

    # Create a thread to handle the generation of the response for this user
    def generate_response_thread():
        with model_lock:
            # Generate the response for this user
            output = model.generate(
                tokenizer.encode(prompt_template, return_tensors="pt", max_length=512),
                attention_mask=torch.ones_like(tokenizer.encode(prompt_template, return_tensors="pt", max_length=512), dtype=torch.long),
                max_length=50,
                num_beams=4,
                early_stopping=True,
            )

        # Convert the generated text back to a string
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    # Start the thread
    response_thread = threading.Thread(target=generate_response_thread)
    response_thread.start()

    # Wait for the thread to finish
    response_thread.join()

    # Return the generated response as JSON
    return jsonify({'response': prompt_template + output[0]})

# Start the API server
if __name__ == '__main__':
    app.run(debug=True, port=8000)
