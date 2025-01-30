import gradio as gr
import torch
from train.model import GPT
from train.config import GPTConfig

class ModelService:
    def __init__(self, checkpoint_path):
        # 加载配置和模型
        self.config = GPTConfig()
        self.model = GPT(self.config)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def generate_text(self, prompt, max_tokens=100):
        # 将提示文本转换为 token
        input_ids = torch.tensor([[self.tokenizer.encode(prompt)]])
        
        # 生成文本
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens)
            
        # 解码输出
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        return generated_text

def create_web_ui():
    # 初始化模型服务
    model_service = ModelService("path/to/your/checkpoint.pt")
    
    # 创建界面
    iface = gr.Interface(
        fn=model_service.generate_text,
        inputs=[
            gr.Textbox(label="输入提示文本"),
            gr.Slider(minimum=10, maximum=500, value=100, label="生成长度")
        ],
        outputs=gr.Textbox(label="生成结果"),
        title="GPT 文本生成",
        description="输入提示文本，生成相关内容"
    )
    
    # 启动服务
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    create_web_ui()