import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import AutoModel, AutoTokenizer

# model_dir = '/opt/app-root/src/granite-7b-lab'
model_dir = './models/granite-7b-lab'

model = AutoModel.from_pretrained(model_dir).cuda()  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

# Prepare a sample input for profiling
instruction_prompts = [
    "Summarize the key points of a recent scientific study on climate change.",
    "Explain the process of photosynthesis in simple terms.",
    "Translate the following sentence to French: 'The quick brown fox jumps over the lazy dog.'",
    "Generate a short story about a dragon who befriends a knight.",
    "Provide a list of five health benefits of regular exercise.",
]
# Profiling on GPU with detailed options
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Include CUDA (GPU) activity
    record_shapes=True,         # Record tensor shapes for more detail
    profile_memory=True,        # Track memory usage
    with_stack=True,            # Capture the call stack
    on_trace_ready=tensorboard_trace_handler('./log')  # Log to TensorBoard
) as prof:
    for i in range(5):  # Run multiple profiling rounds with different tasks
        # Cycle through the instruction prompts for each iteration
        input_text = instruction_prompts[i % len(instruction_prompts)]
        
        # Prepare input for the model and move it to GPU
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)  # Model and inputs are now both on GPU
        torch.cuda.synchronize()  # Ensure all GPU operations are complete before next step
        prof.step()  # Mark each iteration for detailed profiling

# Print profiler summary to console
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
