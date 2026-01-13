from sympy import tensor
from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from torch.nn.utils import rnn
from .mib import mib

# ... (StoppingCriteriaSub, build_one_instance, process_batch_instance 保持不变) ...
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = '</Img> ' + turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) 
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

PROMPT_START = '### Human: <Img>'

class OpenLLAMAPEFTModel(nn.Module):
    '''LoRA for LLaMa model'''
    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        delta_ckpt_path = args.get('delta_ckpt_path') # [FIX] 使用 get 防止报错
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = \
        imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
        
        # =================================================================
        # [FIX] 核心修改：加载 PandaGPT 的预训练 Delta 权重 (LoRA + Projector)
        # =================================================================
        if delta_ckpt_path is not None:
            print(f'Loading pretrained delta weights from {delta_ckpt_path} ...')
            # 假设 delta_ckpt_path 指向 pytorch_model.pt 文件
            delta_state_dict = torch.load(delta_ckpt_path, map_location='cpu')
            
            # 加载 weights，设置 strict=False 因为 delta 只有部分参数
            load_result = self.load_state_dict(delta_state_dict, strict=False)
            print(f"Delta weights loaded. Result: {load_result}")
            
            # 验证 Projector 是否被正确加载 (防止 key 不匹配)
            # PandaGPT 的权重 key 通常是 'llama_proj.weight', 'llama_proj.bias'
            if 'llama_proj.weight' not in delta_state_dict:
                print("WARNING: llama_proj.weight not found in delta checkpoint! Projector is random!")
        else:
            print('No delta_ckpt_path provided. Training from scratch.')
        # =================================================================

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()

    # ... (encode_video, encode_audio, encode_thermal, encode_image, prompt_wrap, forward 等方法保持不变) ...
    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION] 
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1) 
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) 
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO] 
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) 
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) 
        return inputs_llama, atts_llama

    def encode_thermal(self, thermal_paths):
        inputs = {ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['thermal'] 
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) 
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) 
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'] 
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) 
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) 
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(self.device) 
        target_ids = target_ids.to(self.device) 
        attention_mask = attention_mask.to(self.device) 

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) 
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) 
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id 
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) 
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1) 

        empty_targets = (
            torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], 
                       dtype=torch.long).to(self.device).fill_(-100)  
        ) 
        targets = torch.cat([empty_targets, target_ids], dim=1) 
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], dtype=torch.long).to(self.device) 
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() 
        return inputs_embeds, targets, attention_mask 

    def forward(self, inputs):
        image_paths = inputs['image_paths']
        img_embeds, _ = self.encode_image(image_paths)

        output_texts = inputs['output_texts']
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]   
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask   
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    # ... extract_multimodal_feature, prepare_generation_embedding, generate ...
    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        if len(inputs['modality_embeds']) == 1:
            feature_embeds = inputs['modality_embeds'][0]
        else:
            feature_embeds = self.extract_multimodal_feature(inputs)
            inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) 
        text = '</Img> ' + prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) 
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id 
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) 
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1) 
        return inputs_embeds

    def generate(self, inputs):
        input_embeds = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text


class MCPandaModel(OpenLLAMAPEFTModel):
    def __init__(self, **args):
        super().__init__(**args)

        hidden_size = self.llama_model.config.cross_attention_hidden_size
        mib_dim = 128
        beta = 1e-3

        self.aux_head_image = mib(hidden_size, mib_dim, beta)
        self.aux_head_audio = mib(hidden_size, mib_dim, beta)
        # [FIX] 补充 Text 辅助头，否则 forward_aux 会报错
        self.aux_head_text = mib(hidden_size, mib_dim, beta)
        
        self.to(self.device)

    def get_text_embedding(self,input_ids):
        with torch.no_grad():
           embeds = self.llama_model.model.model.embed_tokens(input_ids)
        return embeds.mean(dim=1)

    def forward_main(self,inputs):
        # 1. 获取图像特征
        image_paths = inputs['image_paths']
        if image_paths and image_paths[0] is not None:
            img_embeds, _ = self.encode_image(image_paths)
        else:
            img_embeds, _ = self.encode_image(image_paths) 

        # 2. 获取音频特征
        audio_paths = inputs.get('audio_paths')
        audio_embeds = None
        if audio_paths and audio_paths[0] is not None:
            audio_embeds, _ = self.encode_audio(audio_paths)

        # 3. 特征融合
        if audio_embeds is not None:
            multimodal_embeds = img_embeds + audio_embeds
        else:
            multimodal_embeds = img_embeds

        # 4. 文本处理
        output_texts = inputs['output_texts']
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )
        
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            multimodal_embeds, input_ids, target_ids, attention_mask
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        return outputs.loss

    def forward_aux(self, inputs):
        labels = inputs['labels'].to(self.device)
        losses = {}

        # --- Image Aux ---
        if inputs['image_paths'] and inputs['image_paths'][0] is not None:
            img_embeds, _ = self.encode_image(inputs['image_paths']) 
            img_feat = img_embeds.squeeze(1)
            _, _, loss_v = self.aux_head_image(img_feat, labels)
            losses['loss_v'] = loss_v
        else:
            losses['loss_v'] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Audio Aux ---
        if inputs['audio_paths'] and inputs['audio_paths'][0] is not None:
            aud_embeds, _ = self.encode_audio(inputs['audio_paths'])
            aud_feat = aud_embeds.squeeze(1)
            _, _, loss_a = self.aux_head_audio(aud_feat, labels)
            losses['loss_a'] = loss_a
        else:
            losses['loss_a'] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Text Aux ---
        output_texts = inputs['output_texts']
        input_ids, _, _ = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )
        input_ids = input_ids.to(self.device)
        text_feat = self.get_text_embedding(input_ids)
        
        _, _, loss_t = self.aux_head_text(text_feat, labels)
        losses['loss_t'] = loss_t

        return losses