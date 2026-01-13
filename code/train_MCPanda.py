import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# 确保能导入同目录下的模块
import sys
sys.path.append(os.getcwd())

from header import *
from datasets.mcp_dataset import MCPSupervisedDataset
from model.openllama import MCPandaModel
from min_norm_solvers import MinNormSolver

def parser_args():
    parser = argparse.ArgumentParser(description='KAN-MCP Panda Training')
    parser.add_argument('--data_path', type=str, required=True, help="Path to mosi csv or json")
    parser.add_argument('--image_root_path', type=str, default='')
    parser.add_argument('--audio_root_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--imagebind_ckpt_path', type=str, required=True)
    parser.add_argument('--vicuna_ckpt_path', type=str, required=True)
    parser.add_argument('--delta_ckpt_path', type=str, default=None)
    
    # Training Hyperparams
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_tgt_len', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0, help="Gradient scaling factor from train_mosi")
    
    # Model Config
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--stage', type=int, default=2)
    
    return parser.parse_args()

def flatten_grads(grads):
    """Utility to flatten a list of gradients into a single vector"""
    vec = []
    for g in grads:
        if g is not None:
            vec.append(g.view(-1))
        else:
            # Should not happen if requires_grad is True, but for safety
            pass 
    return torch.cat(vec) if vec else None

def main():
    args = parser_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Initialize Model ---
    print("Initializing KAN-MCP Panda Model...")
    model = MCPandaModel(**vars(args))
    model.train()
    model.to(device)
    
    # Identify parameters
    # shared_params: The projection layer where modalities fuse/conflict
    shared_params = list(model.llama_proj.parameters())
    
    # trainable_params: All params that need optimization (LoRA + Proj + Heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    print(f"Total Trainable params: {len(trainable_params)}")
    print(f"Shared Params (Projector): {len(shared_params)}")

    # --- 2. Dataset ---
    print(f"Loading Dataset from {args.data_path}...")
    dataset = MCPSupervisedDataset(
        data_path=args.data_path,
        image_root_path=args.image_root_path,
        audio_root_path=args.audio_root_path,
        mode='train' 
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate,
        num_workers=4
    )

    # --- 3. Training Loop with MMPareto ---
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0
        
        for step, batch in enumerate(pbar):
            # Move batch to device (done inside model mostly, but labels need help)
            # Labels handling is done inside forward_aux usually, but let's ensure safety if needed
            
            # --- Forward Pass ---
            # 1. Main Task Loss
            loss_main = model.forward_main(batch)
            
            # 2. Aux Task Losses
            losses_aux = model.forward_aux(batch)
            loss_v = losses_aux['loss_v']
            loss_a = losses_aux['loss_a']
            loss_t = losses_aux['loss_t'] # Text Aux
            
            # --- MMPareto Logic (Focus on Shared Projector) ---
            # We want to find the best direction for `llama_proj` to satisfy Main, Visual, Audio.
            # Text Aux largely relies on Embedding layer, which might be frozen or separate, 
            # but we include it if it affects shared params.
            
            # 1. Calculate individual gradients w.r.t Shared Params (llama_proj)
            # We use retain_graph=True because we need to backward multiple times
            
            grads_list = []
            tasks = []
            
            # Helper to get grad vector for shared params
            def get_shared_grad(loss_val):
                if loss_val.requires_grad and loss_val.item() != 0.0:
                    gs = torch.autograd.grad(loss_val, shared_params, retain_graph=True, allow_unused=True)
                    # Replace None with zeros if any
                    gs_clean = []
                    for g, p in zip(gs, shared_params):
                        if g is None: gs_clean.append(torch.zeros_like(p))
                        else: gs_clean.append(g)
                    return flatten_grads(gs_clean)
                return None

            g_main_vec = get_shared_grad(loss_main)
            g_v_vec = get_shared_grad(loss_v)
            g_a_vec = get_shared_grad(loss_a)
            
            # Collect valid gradients for solver
            if g_main_vec is not None:
                grads_list.append(g_main_vec)
                tasks.append('main')
            if g_v_vec is not None:
                grads_list.append(g_v_vec)
                tasks.append('visual')
            if g_a_vec is not None:
                grads_list.append(g_a_vec)
                tasks.append('audio')
                
            # Solver weights initialization (Default: equal weighting or purely Main)
            # If no conflict, we just sum them (weights=1.0) or average.
            # train_mosi Logic: If Cosine > 0, weights = 0.5/0.5 (Average). Else MinNorm.
            
            # Simplified robust logic:
            # If we have multiple tasks active, calculate MinNorm weights to get a conflict-free direction.
            final_shared_grad = None
            
            if len(grads_list) > 1:
                grads_np = [g.detach().cpu().numpy() for g in grads_list]
                weights, _ = MinNormSolver.find_min_norm_element(grads_np)
                weights = torch.tensor(weights, device=device, dtype=torch.float)
                
                # Compute Pareto Gradient
                pareto_grad = torch.zeros_like(grads_list[0])
                for w, g in zip(weights, grads_list):
                    pareto_grad += w * g
                
                # --- Norm Rescaling (Crucial from train_mosi) ---
                # Restore the magnitude of the gradient to avoid vanishing gradients due to conflict cancellation.
                # Reference: train_mosi.py logic
                # We use the norm of the 'standard sum' (or just Main) as the target magnitude.
                
                # Let's use the Sum Gradient as the reference for magnitude
                grad_sum = torch.zeros_like(grads_list[0])
                for g in grads_list:
                    grad_sum += g
                    
                orig_norm = torch.norm(grad_sum)
                pareto_norm = torch.norm(pareto_grad)
                
                # Scale
                if pareto_norm > 0:
                    scale = (orig_norm / pareto_norm) * args.gamma
                    # If scale is too huge (instability), maybe clip it? train_mosi uses diff > 1 check.
                    if scale > 1.0: # Only scale up, or strictly follow train_mosi?
                        # train_mosi: if diff > 1: param.grad = diff * new_grad
                        final_shared_grad = scale * pareto_grad
                    else:
                        final_shared_grad = pareto_grad * args.gamma
                else:
                    final_shared_grad = grad_sum # Fallback
            
            elif len(grads_list) == 1:
                final_shared_grad = grads_list[0]
            
            # --- Final Backward & Override ---
            optimizer.zero_grad()
            
            # Standard Backward to populate gradients for ALL parameters (LoRA, Heads, Projector)
            # We simply sum losses here. The Solver logic only refines the Shared Projector direction.
            # LoRA and Heads will update based on this sum.
            total_loss = loss_main + loss_v + loss_a + loss_t
            total_loss.backward()
            
            # **Override** gradient for Shared Params (llama_proj) with our calculated Pareto Gradient
            if final_shared_grad is not None:
                offset = 0
                for p in shared_params:
                    numel = p.numel()
                    g_segment = final_shared_grad[offset : offset + numel].view_as(p)
                    if p.grad is not None:
                        p.grad.data.copy_(g_segment) # In-place update
                    else:
                        p.grad = g_segment
                    offset += numel

            # Step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += total_loss.item()
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.2f}",
                "L_Main": f"{loss_main.item():.2f}",
                "L_V": f"{loss_v.item():.2f}",
                "L_A": f"{loss_a.item():.2f}"
            })
            
    # Save
    if args.save_path:
        print(f"Saving model to {args.save_path}")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # Save LoRA and Projector
        model.llama_model.save_pretrained(args.save_path)
        # Save Projector specifically if needed, or rely on torch.save for the whole delta
        torch.save(model.llama_proj.state_dict(), os.path.join(args.save_path, "llama_proj.bin"))
        # Save MIB heads
        torch.save({
            'aux_head_image': model.aux_head_image.state_dict(),
            'aux_head_audio': model.aux_head_audio.state_dict(),
            'aux_head_text': model.aux_head_text.state_dict(),
        }, os.path.join(args.save_path, "mib_heads.bin"))

if __name__ == "__main__":
    main()