from header import *
from datasets import *
from model import *
from config import *
from min_norm_solvers import MinNormSolver
def parser_args():
    parser = argparse.ArgumentParser(description='KAN-MCP Panda Training')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_root_path', type=str, default='')
    parser.add_argument('--audio_root_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--imagebind_ckpt_path', type=str, required=True)
    parser.add_argument('--vicuna_ckpt_path', type=str, required=True)
    parser.add_argument('--delta_ckpt_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_tgt_len', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--stage', type=int, default=2)
    return parser.parse_args()

def main():
    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型 (使用新定义的子类)
    print("Initializing KAN-MCP Panda Model...")
    model_args = vars(args)
    model = KANMCPPandaModel(**model_args)
    model.train() # 确保在训练模式
    
    # 2. 优化器
    # 只优化 requires_grad 的参数 (LoRA, Projector, MIB Heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"Trainable params: {len(trainable_params)}")

    # 3. 数据集
    print("Loading Dataset...")
    dataset = KANMCPSupervisedDataset(
        data_path=args.data_path,
        image_root_path=args.image_root_path,
        audio_root_path=args.audio_root_path
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate
    )
    # 4. 训练循环 (MMPareto)
    print("Starting Training with MMPareto Gradient Adjustment...")

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            loss_main = model.forward_main(batch)
            losses_aux = model.forward_aux(batch)
            
            # --- Gradient Computation ---
            # 我们需要分别为每个任务计算梯度，但不立即更新
            
            # 辅助函数：计算梯度并扁平化
            def compute_grad_vec(loss):
                if not loss.requires_grad or loss.item() == 0.0:
                    return None
                
                # retain_graph=True 因为要多次 backward
                grads = torch.autograd.grad(
                    loss, trainable_params, retain_graph=True, allow_unused=True
                )
                
                # 扁平化所有梯度
                grad_vec = []
                for g, p in zip(grads, trainable_params):
                    if g is not None:
                        grad_vec.append(g.view(-1))
                    else:
                        grad_vec.append(torch.zeros_like(p).view(-1))
                return torch.cat(grad_vec)

            # 获取梯度向量
            g_main = compute_grad_vec(loss_main)
            g_v = compute_grad_vec(losses_aux['loss_v'])
            g_a = compute_grad_vec(losses_aux['loss_a'])
            g_t = compute_grad_vec(losses_aux['loss_t'])
            
            # 过滤掉 None (比如某个 batch 没有音频)
            valid_grads = []
            valid_losses = [] # 仅用于记录日志
            
            if g_main is not None: 
                valid_grads.append(g_main)
                valid_losses.append(loss_main.item())
            if g_v is not None: valid_grads.append(g_v)
            if g_a is not None: valid_grads.append(g_a)
            if g_t is not None: valid_grads.append(g_t)

            if not valid_grads:
                continue

            # --- MMPareto Adjustment ---
            if len(valid_grads) > 1:
                # 转换为 numpy 列表供 solver 使用
                grads_np = [g.detach().cpu().numpy() for g in valid_grads]
                
                # 计算 Pareto 权重
                sol, _ = MinNormSolver.find_min_norm_element(grads_np)
                weights = torch.tensor(sol, device=device, dtype=torch.float)
                
                # 加权合并梯度
                final_grad_vec = torch.zeros_like(valid_grads[0])
                for w, g in zip(weights, valid_grads):
                    final_grad_vec += w * g
            else:
                # 只有一个任务有梯度，直接用
                final_grad_vec = valid_grads[0]

            # --- Apply Gradients ---
            # 将合并后的梯度赋值回 parameters.grad
            offset = 0
            for p in trainable_params:
                numel = p.numel()
                # 提取对应参数的梯度片段
                g_segment = final_grad_vec[offset : offset + numel]
                if p.grad is None:
                    p.grad = g_segment.view_as(p)
                else:
                    p.grad += g_segment.view_as(p)
                offset += numel

            # 更新参数
            optimizer.step()
            
            # 释放 graph
            # 注意：在最后一次 backward 后，graph 通常会自动释放，
            # 但这里我们是手动 autograd.grad，需要确保没内存泄漏。
            # 简单的做法是：本轮结束，变量重置即可。

            # 日志
            pbar.set_postfix({
                "L_Main": f"{loss_main.item():.2f}",
                "L_Vis": f"{losses_aux['loss_v'].item():.2f}",
                "L_Aud": f"{losses_aux['loss_a'].item():.2f}"
            })

    # 保存模型
    if args.save_path:
        print(f"Saving model to {args.save_path}...")
        model.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
