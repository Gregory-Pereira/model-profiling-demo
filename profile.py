import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import AutoModel, AutoTokenizer

# Path to the Hugging Face model directory
model_dir = './path_to_your_model_directory'

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

# Prepare a sample input for profiling
input_text = "This is a sample input to profile the model."
inputs = tokenizer(input_text, return_tensors="pt")  # Convert text to tensor input

# model = torch.load('model.pth')
model.eval()  # Set to evaluation mode for inference profiling

# Create a dummy input matching the model's expected input size
inputs = torch.randn(1, 3, 224, 224)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    on_trace_ready=tensorboard_trace_handler('./log')
) as prof:
    for _ in range(5):  # Run multiple inference steps for profiling
        outputs = model(inputs)
        prof.step()  # Mark each iteration

# Print summarized profiling information to the console
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
