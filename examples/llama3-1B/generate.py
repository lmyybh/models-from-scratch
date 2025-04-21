import argparse

from ...llama3 import LlamaModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using llama3"
    )
    parser.add_argument("--ckpt", type=str, required=True, help='Location of the "LLM-Research/Llama-3.2-1B" checkpoint')
    parser.add_argument('--max', type=int, default=128, help='max length of the text')
    args = parser.parse_args()
    
    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    llama3 = LlamaModel.build_1B(
        ckpt_file=args.ckpt
    )
    
    print("-" * 100)
    print("Model:", llama3.config.model_name)
    print("Max Length:", args.max)
    print("-" * 100)
    
    prompt = input("[Please input a prompt]: ")
    output = llama3.generate(prompt, max_length=args.max)
    print(output)
