import gradio as gr
from huggingface_hub import InferenceClient
import torch
import time
import psutil  # Import psutil to track CPU memory usage
from transformers import pipeline
from prometheus_client import start_http_server, Counter, Summary

# Prometheus metrics
REQUEST_COUNTER = Counter('app_requests_total', 'Total number of requests')
SUCCESSFUL_REQUESTS = Counter('app_successful_requests_total', 'Total number of successful requests')
FAILED_REQUESTS = Counter('app_failed_requests_total', 'Total number of failed requests')
REQUEST_DURATION = Summary('app_request_duration_seconds', 'Time spent processing request')

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

def respond(
    message,
    history: list[tuple[str, str]],
    system_message="You are a friendly chatbot who always responds in the style of a therapist.",
    max_tokens=600,
    temperature=0.6,
    top_p=0.65,
    use_local_model=False,
):
    
    global stop_inference
    stop_inference = False  # Reset cancellation flag
    REQUEST_COUNTER.inc()  # Increment request counter
    request_timer = REQUEST_DURATION.time()  # Start timing the request

    start_time = time.time()  # Start time tracking
    process = psutil.Process()
    initial_memory = process.memory_info().rss  # Memory before in bytes

    try:
        # Initialize history if it's None
        if history is None:
            history = []
        
        if use_local_model:
            # local inference 
            messages = [{"role": "system", "content": system_message}]
            for val in history:
                if val[0]:
                    messages.append({"role": "user", "content": val[0]})
                if val[1]:
                    messages.append({"role": "assistant", "content": val[1]})
            messages.append({"role": "user", "content": message})
            
            response = ""
            
            for output in pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            ):
                if stop_inference:
                    response = "Inference cancelled."
                    yield history + [(message, response)]
                    return
                token = output['generated_text'][-1]['content']
                response += token
                yield history + [(message, response)]  # Yield history + new response
                
        else:
            # API-based inference 
            messages = [{"role": "system", "content": system_message}]
            for val in history:
                if val[0]:
                    messages.append({"role": "user", "content": val[0]})
                if val[1]:
                    messages.append({"role": "assistant", "content": val[1]})
            messages.append({"role": "user", "content": message})
            
            response = ""
            for message_chunk in client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
            ):
                if stop_inference:
                    response = "Inference cancelled."
                    yield history + [(message, response)]
                    return
                if stop_inference:
                    response = "Inference cancelled."
                    break
                
                token = message_chunk.choices[0].delta.content
                response += token
                yield history + [(message, response)]  # Yield history + new response
            
            SUCCESSFUL_REQUESTS.inc()  # Increment successful request counter
    except Exception as e:
        FAILED_REQUESTS.inc()  # Increment failed request counter
        yield history + [(message, f"Error: {str(e)}")]
    finally:
        request_timer.observe_duration()  # Stop timing the request

    # Calculate elapsed time after response generation
    end_time = time.time()
    final_memory = process.memory_info().rss # Memory usage i
    memory_used = final_memory - initial_memory
    memory_in_mb = memory_used/1048576
    elapsed_time = end_time - start_time

    # Append the memory usage and elapsed time to the response
    final_response = f"{response}\n\n(Generated in {elapsed_time:.2f} seconds, Memory used: {memory_in_mb:.6f} MB)"
    
    yield history + [(message, final_response)]  # Yield final response with elapsed time

def cancel_inference():
    global stop_inference
    stop_inference = True

# Custom CSS for a fancy look
custom_css = """
#main-container {
    background-color: aquamarine;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: aquamarine;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gradio-button {
    background-color: red;
    color: blue;
    border: none;
    border-radius: 36px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: aquamarine;
}
.gr-slider {
    color: aquamarine;
}
.gr-chat {
    font-size: 23px;
    background: aquamarine;
}
#title {
    text-align: center;
    font-size: 6em;
    margin-bottom: 20px;
    color: aquamarine;
}
.halt-button {
    background-color: red;
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
}
.halt-button:hover {
    background-color: darkred;
}
.submit-button {
    background-color: red;
    color: black;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
}
.submit-button:hover {
    background-color: darkgreen;
}
"""

# Define the interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🍍 NORA: Nutrition Optimization and Recommendation Assistant 🍎</h1>")
    gr.Markdown("# 🍓 AI-driven Nutritionist (Product Demo)\nThis personal nutritionist is based on Zephyr-7b-beta (Hugging Face API-based inference) as well as Phi-3-mini-4k-instruct (Microsft's local inference used for local inference). Interact with NORA using the customizable settings below, describe your nutritional needs, and let our AI assistant guide you!")

    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly chatbot who always responds in the style of a professional nutritionist.", label="NORA's System message", interactive=False)
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=600, step=1, label="Max new tokens (controls the length of the response)", elem_classes="gr-slider")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.6, step=0.1, label="Temperature (affects the creativity and randomness of the generated response)", elem_classes="gr-slider")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.65, step=0.05, label="Top-p (nucleus sampling - balances diversity and coherence in token selection)", elem_classes="gr-slider")

    gr.Markdown("### Model Output 👇")

    chat_history = gr.Chatbot(label="NORA's response below")

    user_input = gr.Textbox(show_label=False, placeholder="Message NORA here...")

    with gr.Row():
        submit_button = gr.Button("Submit", elem_classes="submit-button")  # Add submit button
        cancel_button = gr.Button("Halt!", variant="danger", elem_classes="halt-button")

    # Adjusted to ensure history is maintained and passed correctly
    submit_button.click(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)
    user_input.submit(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)

    cancel_button.click(cancel_inference)

    gr.Markdown("# Disclaimer:\nNORA is designed to provide general nutritional guidance and personalized meal suggestions based on the information you provide. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a licensed healthcare provider or nutritionist before making significant changes to your diet or addressing specific health concerns. NORA’s recommendations are based on AI algorithms and user input, and while we strive for accuracy, results may vary. Use NORA responsibly and in conjunction with professional guidance as needed. By using this app, you agree that NORA is not liable for any health outcomes or decisions made based on its recommendations.")

if __name__ == "__main__":
    start_http_server(8000)  # Expose metrics on port 8000
    demo.launch(share=False)  # Remove share=True because it's not supported on HF Spaces