import torch
import gradio as gr
from model import GPT
from config import GPTConfig
import os
from transformers import BertTokenizer  # 添加导入

class TextGenerator:
    def __init__(self, checkpoint_path):
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 初始化模型配置
        self.model_config = GPTConfig(vocab_size=21128)  # BERT中文词表大小
        
        # 初始化模型
        self.model = GPT(self.model_config)
        
        # 加载检查点
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = state_dict[key]
            self.model.load_state_dict(new_state_dict)
            print(f"模型已从 {checkpoint_path} 加载")
        else:
            raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50):
        # 使用BERT分词器编码输入文本
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True)
        ).unsqueeze(0).to(self.device)
        
        # 生成文本
        for _ in range(max_length):
            outputs = self.model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 如果生成了[SEP]标记，则停止生成
            if next_token.item() == self.tokenizer.sep_token_id:
                break
        
        # 使用BERT分词器解码生成的文本
        generated_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        return generated_text

def create_interface(checkpoint_path):
    # 初始化生成器
    generator = TextGenerator(checkpoint_path)
    
    # 创建Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("# GPT文本生成器")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="输入提示", lines=3)
                max_length = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="最大生成长度")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="温度")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
                generate_btn = gr.Button("生成")
            
            with gr.Column():
                output = gr.Textbox(label="生成的文本", lines=8)
        
        generate_btn.click(
            fn=generator.generate,
            inputs=[prompt, max_length, temperature, top_k],
            outputs=output
        )
        
        gr.Markdown("""
        ## 使用说明
        - 在输入框中输入提示文本
        - 调整生成参数：
          - 最大生成长度：控制生成文本的最大长度
          - 温度：控制生成文本的随机性，值越大越随机
          - Top-K：控制每一步生成时考虑的候选词数量
        - 点击"生成"按钮开始生成文本
        """)
    
    return demo

if __name__ == "__main__":
    # 设置检查点路径
    checkpoint_path = "checkpoints/default_run_step_500.pt"  # 根据实际保存的检查点文件名修改
    
    # 创建并启动界面
    demo = create_interface(checkpoint_path)
    demo.launch(share=True)  # share=True 允许生成公共URL访问 