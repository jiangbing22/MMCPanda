import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from KANMCP import KANMCP

import global_configs
from global_configs import DEVICE

import warnings
from min_norm_solvers import MinNormSolver
import torch.nn.functional as F
import re


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=60)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=35)
parser.add_argument("--text_learning_rate", type=float, default=1e-5)
parser.add_argument("--other_learning_rate", type=float, default=1e-3)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1024)
parser.add_argument("--m_dim", type=int, default=1024)

parser.add_argument('--kan_hidden_neurons', type=int, default=4)
parser.add_argument('--compressed_dim', type=int, default=3)

parser.add_argument('--weight1', type=float, default=0.1)
parser.add_argument('--weight2', type=float, default=1)

parser.add_argument('--gamma', type=float, default=1.5)

parser.add_argument('--tqdm_disable', type=bool, default=False)
parser.add_argument('--use_MMPareto', type=bool, default=True)

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM,global_configs.TEXT_DIM)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # print("Seed: {}".format(seed))


def prep_for_training(num_train_optimization_steps: int):
    model = KANMCP.from_pretrained(args.model, multimodal_config=args, num_labels=1)

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    text_parameters = []
    other_parameters = []

    for item in param_optimizer:
        if "Deberta" in item[0].split("."):
            text_parameters.append(item[0])
        elif "TEncoder" in item[0].split("."):
            text_parameters.append(item[0])
        else:
            other_parameters.append(item[0])

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in text_parameters)
            ],
            "weight_decay": 0.01,
            "lr": args.text_learning_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in other_parameters)
            ],
            "weight_decay": 0.01,
            "lr": args.other_learning_rate,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    """
    在一个训练周期（epoch）内训练模型。

    Args:
        model (nn.Module): 要训练的 PyTorch 模型。
        train_dataloader (DataLoader): 提供训练数据的 DataLoader。
        optimizer: 用于更新模型参数的优化器。
        scheduler: 学习率调度器。

    Returns:
        float: 当前 epoch 的平均训练损失。
    """
    model.train()  # 将模型设置为训练模式
    tr_loss = 0  # 累积总训练损失
    tr_audio_loss = 0  # 累积音频模态损失
    tr_visual_loss = 0  # 累积视觉模态损失
    tr_text_loss = 0  # 累积文本模态损失
    nb_tr_examples, nb_tr_steps = 0, 0  # 训练样本数和训练步数计数器

    # 用于记录不同模态编码器参数的列表
    record_names_audio = []
    record_names_visual = []
    record_names_text = []

    # 遍历模型的所有命名参数，将编码器参数按模态分类
    for name, param in model.named_parameters():
        if 'AEncoder' in name:  # 如果参数属于音频编码器
            if 'decoder' in name:  # 跳过解码器参数
                continue
            record_names_audio.append((name, param))
            continue
        if 'VEncoder' in name:  # 如果参数属于视觉编码器
            if 'decoder' in name:  # 跳过解码器参数
                continue
            record_names_visual.append((name, param))
            continue
        if 'TEncoder' in name:  # 如果参数属于文本编码器
            if 'decoder' in name:  # 跳过解码器参数
                continue
            record_names_text.append((name, param))
            continue
        if 'Deberta' in name:  # 如果参数属于 Deberta 模型（通常是文本编码器的一部分）
            record_names_text.append((name, param))
            continue

    # 遍历训练数据加载器中的每个批次
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.tqdm_disable)):
        # 将批次数据移动到指定设备 (CPU/GPU)
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch  # 解包批次数据

        # 移除视觉和音频特征中维度为1的单例维度
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        # 对视觉和音频特征进行归一化到 [0, 1] 范围
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

        # 模型前向传播
        outputs = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            label_ids,
        )

        logits = outputs["logits"]  # 模型输出的预测结果
        loss_t = outputs["loss_t"]  # 文本模态的辅助损失
        loss_a = outputs["loss_a"]  # 音频模态的辅助损失
        loss_v = outputs["loss_v"]  # 视觉模态的辅助损失

        loss_fct = MSELoss()  # 定义均方误差损失函数，用于主任务

        # 如果启用多模态 Pareto 优化 (MMPareto)
        if args.use_MMPareto:
            # 计算主任务（多模态融合）的损失
            loss_mm = loss_fct(logits.view(-1), label_ids.view(-1))

            # 收集所有损失
            losses = [loss_mm, loss_t, loss_v, loss_a]
            all_loss = ['both', 'text', 'visual', 'audio']  # 损失类型名称

            # 初始化用于存储梯度的字典
            grads_text = {}
            grads_audio = {}
            grads_visual = {}

            # 对每个损失单独进行反向传播并记录梯度
            for idx, loss_type in enumerate(all_loss):
                loss = losses[idx]
                # 反向传播，retain_graph=True 允许后续损失继续使用计算图
                loss.backward(retain_graph=True)

                # 根据损失类型记录对应模态编码器的梯度
                if loss_type == 'visual':
                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        # 克隆梯度数据，防止被后续操作修改
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    # 将视觉模态的所有梯度展平并拼接，用于计算余弦相似度
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])
                elif loss_type == 'audio':
                    for tensor_name, param in record_names_audio:
                        if loss_type not in grads_audio.keys():
                            grads_audio[loss_type] = {}
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])
                elif loss_type == 'text':
                    for tensor_name, param in record_names_text:
                        if loss_type not in grads_text.keys():
                            grads_text[loss_type] = {}
                        grads_text[loss_type][tensor_name] = param.grad.data.clone()
                    grads_text[loss_type]["concat"] = torch.cat(
                        [grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text])
                else:  # 'both' 损失，需要记录所有模态的梯度
                    for tensor_name, param in record_names_text:
                        if loss_type not in grads_text.keys():
                            grads_text[loss_type] = {}
                        grads_text[loss_type][tensor_name] = param.grad.data.clone()
                    grads_text[loss_type]["concat"] = torch.cat(
                        [grads_text[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_text])

                    for tensor_name, param in record_names_audio:
                        if loss_type not in grads_audio.keys():
                            grads_audio[loss_type] = {}
                        grads_audio[loss_type][tensor_name] = param.grad.data.clone()
                    grads_audio[loss_type]["concat"] = torch.cat(
                        [grads_audio[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_audio])

                    for tensor_name, param in record_names_visual:
                        if loss_type not in grads_visual.keys():
                            grads_visual[loss_type] = {}
                        grads_visual[loss_type][tensor_name] = param.grad.data.clone()
                    grads_visual[loss_type]["concat"] = torch.cat(
                        [grads_visual[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_visual])

                optimizer.zero_grad()  # 清零优化器梯度，避免梯度累积到下一个损失的计算中

            # 计算主任务梯度与各模态辅助任务梯度之间的余弦相似度
            this_cos_text = F.cosine_similarity(grads_text['both']["concat"], grads_text['text']["concat"], dim=0)
            this_cos_audio = F.cosine_similarity(grads_audio['both']["concat"], grads_audio['audio']["concat"], dim=0)
            this_cos_visual = F.cosine_similarity(grads_visual['both']["concat"], grads_visual['visual']["concat"],
                                                  dim=0)

            # 定义任务组
            text_task = ['both', 'text']
            audio_task = ['both', 'audio']
            visual_task = ['both', 'visual']

            # 初始化各模态的损失权重
            # text_k[0]: 多模态损失的权重
            # text_k[1]: 文本模态损失的权重
            text_k = [0, 0]
            audio_k = [0, 0]
            visual_k = [0, 0]

            # 根据梯度余弦相似度调整损失权重
            # 如果余弦相似度 > 0，表示梯度方向一致，任务之间冲突较小，使用等权重
            # 否则，使用 MinNormSolver 解决梯度冲突，找到 Pareto 最优权重
            if this_cos_text > 0:
                text_k[0] = 0.5
                text_k[1] = 0.5
            else:
                # 使用 MinNormSolver 寻找最小范数梯度组合的权重
                text_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_text[t].values()) for t in text_task])
                loss_t = args.weight2 * loss_t  # 对文本辅助损失进行加权

            if this_cos_audio > 0:
                audio_k[0] = 0.5
                audio_k[1] = 0.5
            else:
                audio_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_audio[t].values()) for t in audio_task])
                loss_a = args.weight2 * loss_a  # 对音频辅助损失进行加权

            if this_cos_visual > 0:
                visual_k[0] = 0.5
                visual_k[1] = 0.5
            else:
                visual_k, min_norm = MinNormSolver.find_min_norm_element(
                    [list(grads_visual[t].values()) for t in visual_task])
                loss_v = args.weight2 * loss_v  # 对视觉辅助损失进行加权

            # 组合所有损失形成最终的总损失
            loss = loss_mm + args.weight1 * (loss_t + loss_v + loss_a)
            # 如果启用梯度累积，则对损失进行平均
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss.backward()  # 对最终的总损失进行反向传播

            gamma = args.gamma  # 获取 gamma 参数，用于梯度调整
            # 遍历模型参数，根据 MMPareto 策略调整梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer = re.split('[_.]', str(name))  # 解析参数名称以确定所属层
                    # if('head' in layer):
                    #     continue # 忽略某些层（此处被注释）

                    # 文本编码器参数的梯度调整
                    if 'TEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())  # 当前梯度的范数
                        # 根据计算出的权重组合文本相关梯度
                        new_grad = 2 * text_k[0] * grads_text['both'][name] + 2 * text_k[1] * grads_text['text'][name]
                        new_norm = torch.norm(new_grad)  # 新梯度的范数
                        diff = three_norm / new_norm  # 范数比率
                        # 如果范数比率大于1，则按比例放大新梯度，否则直接使用新梯度
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    # Deberta 模型参数的梯度调整（与文本编码器类似）
                    if 'Deberta' in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * text_k[0] * grads_text['both'][name] + 2 * text_k[1] * grads_text['text'][name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    # 音频编码器参数的梯度调整
                    if 'AEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * audio_k[0] * grads_audio['both'][name] + 2 * audio_k[1] * grads_audio['audio'][
                            name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

                    # 视觉编码器参数的梯度调整
                    if 'VEncoder' in layer and 'decoder' not in layer:
                        three_norm = torch.norm(param.grad.data.clone())
                        new_grad = 2 * visual_k[0] * grads_visual['both'][name] + 2 * visual_k[1] * \
                                   grads_visual['visual'][name]
                        new_norm = torch.norm(new_grad)
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma
        else:
            # 如果不使用 MMPareto 优化，则进行标准损失计算和反向传播
            # 主任务损失与辅助损失加权求和
            loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 0.1*(loss_t + loss_v + loss_a)
            # 如果启用梯度累积，则对损失进行平均
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss.backward()  # 反向传播

        # 累积损失值
        tr_loss += loss.item()
        tr_visual_loss += loss_v.item()
        tr_audio_loss += loss_a.item()
        tr_text_loss += loss_t.item()
        nb_tr_steps += 1  # 增加训练步数计数

        # 梯度累积：当达到指定步数时，执行优化器更新和学习率调度
        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()  # 清零优化器梯度

    # 返回当前 epoch 的平均训练损失
    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0

    # 11.30
    preds = []
    labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration", disable=True)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids,
            )

            logits = outputs["logits"]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

            # 11.30
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, disable=True):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                label_ids,
            )

            logits = outputs["logits"]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc2 = accuracy_score(y_test, preds)

    return acc7, acc2, f_score, mae, corr


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
):
    min_eval_loss = 1000
    test_result = {
        "acc2": 0,
        "acc7": 0,
        "f1": 0,
        "mae": 0,
        "corr": 0,
    }

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        eval_loss = eval_epoch(model, validation_dataloader)
        acc7, acc2, f_score, mae, corr = test_score_model(model, test_data_loader)

        # 输出
        print("TRAIN: epoch:{}, train_loss:{}, eval_loss:{}".format(epoch_i + 1, train_loss, eval_loss))
        print("TEST: acc7: {}, acc2: {}, f1: {}, mae: {}, corr: {}".format(acc7, acc2, f_score, mae, corr))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            test_result["acc2"] = acc2
            test_result["acc7"] = acc7
            test_result["f1"] = f_score
            test_result["mae"] = mae
            test_result["corr"] = corr

        if epoch_i + 1 == args.n_epochs:
            print("====RESULT====")
            print("acc7:{}, acc2:{}, f1:{}, mae:{}, corr:{}".format(test_result["acc7"], test_result["acc2"],test_result["f1"], test_result["mae"],test_result["corr"]))

    return test_result


def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    print(args)

    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)
    
    train(
       model,
       train_data_loader,
       dev_data_loader,
       test_data_loader,
       optimizer,
       scheduler,
    )
    return model


if __name__ == '__main__':
    main()