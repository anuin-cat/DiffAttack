import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks
import utils
import imagenet_label


def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0

def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================

            其实就是完整的实现一次DDIM的逆向过程, 然后将逆向过程所有步的latent存起来放到latents
            逆过程的每一步中都插入了text的embeding (而且还是两个, 第一个是空的prompt, 第二个是真实prompt)
            
    """
    batch_size = 1
    max_length = 77

    # 1. 获取无条件文本生成任务的输入嵌入

    # 1.1 使用分词器对空字符串进行处理，生成用于无条件文本生成任务的输入（和.encode输出内容相同，就多了一个mask）
    #     pt表示pytorch，返回tensor格式数据
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    # 1.2 将无条件文本生成任务的输入文本序列传递给文本编码器，获取其嵌入表示
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    # 2. 获取有条件文本生成任务的输入嵌入
    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    # 3. 获取图像的嵌入表示（vae.decode）
    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2) # 因为context有两个，所以这里也要有两个，后边还要将其切分开
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2) # 从维度0切分成两个tensor
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond) # 某种优化操作

        # 训练一共 n 步，推理我要求 k 步就推理完，所以每次推进的步数是 n/k 才符合训练的目的
        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0) # alphas_cumprod就是alpha_bar

        "leverage reversed_x0" # 其实就是那个公式
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents # 迭代最后一步的latents以及中间去噪的所有latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):

        def forward(x, context=None):
            # x原本是4 4 28 28 -> 4 4 784 因为要先做一个卷积，通道增大所以 4 变成了 320 -> 4 784 320
            # x : 4 784 320
            # context : 4 784 320
            q = self.to_q(x) # 4 784 320 （就是换了个shape，数据量没变）
            is_cross = context is not None
            context = context if is_cross else x # 4 784 320 | 4 77 1024
            k = self.to_k(context) # 4 784 320 | 4 77 320
            v = self.to_v(context) # 4 784 320 | 4 77 320
            q = self.reshape_heads_to_batch_dim(q) # 20 784 64
            k = self.reshape_heads_to_batch_dim(k) # 20 784 64 | 20 77 64
            v = self.reshape_heads_to_batch_dim(v) # 20 784 64 | 20 77 64

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale # 20 784 784 | 20 784 77（爱因斯坦求和）

            attn = sim.softmax(dim=-1) # 20 784 784 | 20 784 77
            attn = controller(attn, is_cross, place_in_unet) # 20 784 784 | 20 784 77
            out = torch.einsum("b i j, b j d -> b i d", attn, v) # 20 784 64
            out = self.reshape_batch_dim_to_heads(out) # 4 784 320

            out = self.to_out[0](out) # 4 784 320
            out = self.to_out[1](out) # 4 784 320
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        # place_in_unet 表示 上采样up、下采样down、中间层mid
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet) # 将 CrossAttention 层的正向传播更改为 ca_forward 内定义的函数
            return count + 1
        elif hasattr(net_, 'children'): # 递归查询子网络中是否有 CrossAttention 并注册
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(x, context=None):
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)

            out = self.to_out[0](out)
            out = self.to_out[1](out)
            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2) # 4 4 28 28 -> 4 4 784
    # context：4 77 1024
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.enable_grad()
def diffattack(
        model,
        label, # 真实标签
        controller,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        image=None,
        model_name="inception",
        save_path=r"/home/DiffAttack/output",
        res=224,
        start_step=15,
        iterations=30,
        verbose=True,
        topN=1,
        args=None
):
    label = torch.from_numpy(label).long().cuda() # 表情605

    model.vae.requires_grad_(False) # vae用于将图像压缩到latent
    model.text_encoder.requires_grad_(False) # text_encoder用于对文本进行编码
    model.unet.requires_grad_(False) # unet用于加噪

    # 1. 获取代理模型（605）
    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    # resize后的图像大小
    height = width = res

    # 2. 图像resize + 标准化 + 转换为tensor
    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    # 3. 代理模型预测（label维度只有1，为啥要拿准确度呢）
    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred) # 得到预测的置信度logit
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item()) # 展示预测内容索引 + 标签内容的置信度

    _, pred_labels = pred.topk(topN, largest=True, sorted=True)

    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, topN)]) # label对应的物体描述（iPod）
    prompt = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2 # 将描述乘2
    print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

    # 4. 获取标签文本编码（编码是确定的，不可训练的，能够恢复原始语言的）
    #    而嵌入（embedding）则是会损失信息的，可训练的，不确定的内容，常见的有 Word2Vec、GloVe、BERT
    true_label = model.tokenizer.encode(imagenet_label.refined_Label[label.item()]) # 真实标签 单个的编码
    target_label = model.tokenizer.encode(target_prompt)    # 目标标签 多个的编码（中间含有单个的真实标签）
    print("decoder: ", true_label, target_label)

    """
            ==========================================
            ============ DDIM Inversion ==============
            === Details please refer to Appendix B ===
            ==========================================
    """

    # 5. 通过DDIM获取图像到噪声的 n 步的加噪过程（就是逆向过程）
    #    latent是加噪的最后一步得到的噪声潜空间，inversion_latents是所有噪声潜空间的集合
    #    潜空间指的是通过vae.decode将图像降采样到一个潜空间
    latent, inversion_latents = ddim_reverse_sample(image, prompt, model, # 逆向DDIM（加噪）
                                                    num_inference_steps,
                                                    0, res=height)
    inversion_latents = inversion_latents[::-1] # 逆转latents顺序，本来最前边的是清晰图，现在最前边是噪声图

    init_prompt = [prompt[0]] # 初始的指导文本，用于后期梯度优化
    batch_size = len(init_prompt)
    # 抽取一个加噪后的latents作为图像生成起始点（start_step越大，噪声越小）
    # inversion_latents[0] 是纯噪声端 inversion_latents[-1]是原图端
    latent = inversion_latents[start_step - 1]  # 1 4 28 28
    

    """
            ===============================================================================
            === Good initial reconstruction by optimizing the unconditional embeddings ====
            ======================= Details please refer to Section 3.4 ===================
            ===============================================================================
    """

    # 6. 将空文本（无条件）编码为token然后再嵌入为embedding
    #    我们可以通过 tokenizer.pad_token_id 得到填充的内容是什么，进而得到真实编码的长度是多少
    #    后边的真实标签的注意力的选取似乎也和这个有关系
    max_length = 77
    uncond_input = model.tokenizer( # 和 model.tokenizer.encode 作用相同，不过这里给的是一个map，input_ids是和其相同的内容
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    ) # 1 77
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0] # 嵌入 1 77 1024

    # 6. 将prompt 目标标签 编码为token然后再嵌入为embedding
    #    tokenizer() 与 tokenizer.encode() 是进行编码的，text_encoder是进行嵌入的
    text_input = model.tokenizer(
        init_prompt, # 单个初始指导文本，用于后期梯度优化作为攻击源
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    # 初始化latents，就是更改了一下latent的大小（size），扩展了一下维度（规范化而已，其实Latent已经是这么大了）
    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size) # latent, latents size相同 latent = size/8

    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1) # 只对uncond_embeddings进行优化，也就是空文本的嵌入
    loss_func = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
    # 7. 通过更新 uncond_embeddings 来确保拼合以后的 context 能够达成和 text_embeddings 类似的指导效果
    for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
        for _ in range(10 + 2 * ind): # 迭代优化 k 次
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1]) # 拉近真实噪声和无条件噪声的距离，来优化uncond嵌入
            loss.backward()
            optimizer.step()

            context = [uncond_embeddings, text_embeddings]
            context = torch.cat(context)

        with torch.no_grad(): # 相当于重复进行了一下上边的内容（但是为什么不直接用上边的呢？）
            # 这里有点疑惑，为什么要分批存储，并且在下边攻击的时候分批使用（难道是不同阶段对无条件嵌入的依赖极强？）
            latents = diffusion_step(model, latents, context, t, guidance_scale).detach() # 推进latents去噪一步（新方法）
            all_uncond_emb.append(uncond_embeddings.detach().clone()) # 将优化过程中所有uncond_embedding保存下来

    """
            ==========================================
            ============ Latents Attack ==============
            ==== Details please refer to Section 3 ===
            ==========================================

            这里的目标就是对latent进行优化, 让其去噪以后具备攻击能力
    """

    uncond_embeddings.requires_grad_(False)
    
    register_attention_control(model, controller) # 注册控制器
    batch_size = len(prompt)

    # 8. 将 prompt 编码为 token 并获取其 embedding
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ) # 2 77
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0] # 2 77 1024

    # 9. 拼接 embedding
    #    将前边得到的 uncond_embeddings * 2 和 text_embeddings 拼接起来（一共有5个 uncond_embeddings，所以拼五份）
    #    因为 text_embeddings 是 2 77 1024，所以两者拼接以后为 4 77 1024
    context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
    context = [torch.cat(i) for i in context] # [4, 77, 1024] * 5

    original_latent = latent.clone() # 1 4 28 28

    latent.requires_grad_(True) # 优化潜变量latent

    # 10. 设置 latent 的优化器以及损失函数
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    # “Pseudo” Mask for better Imperceptibility, yet sacrifice the transferability. Details please refer to Appendix D.
    # 11. 使用mask让攻击更加隐蔽，但是会牺牲可迁移性
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    # 12. 开始优化latent（主 攻击模块）
    pbar = tqdm(range(iterations), desc="Iterations")
    for _, _ in enumerate(pbar): # attack iter
        controller.loss = 0

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        # 13. 从选定步数开始进行去噪，拿到去噪完成后的 latent
        controller.reset()
        latents = torch.cat([original_latent, latent]) # [2 4 28 28] original_latent = latent.clone()
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            # 循环中 controller 被一同执行获取到 Unet 中的注意力图
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale) # 经典的 DDIM 去噪

        # aggregate：总；获取总注意力图 map before和after表示 [0] [1]
        # 14. 获取注意力图像
        #     after 表示攻击以后，before 表示原图的注意力图像
        #     这里取 before_true_label_attention_map 时只取中间部分可能是因为取完的话导致效果太极端
        before_attention_map = aggregate_attention(prompt, controller, 7, ("up", "down"), True, 0, is_cpu=False) # 7 7 77
        after_attention_map = aggregate_attention(prompt, controller, 7, ("up", "down"), True, 1, is_cpu=False) # 7 7 77
        before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1] # 难道和嵌入长度有关
        after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1] # 7 7 2 最后一个维度表示每个嵌入对图像的注意力，所以后边空嵌入就没意义了

        # 15. 获取mask（根据注意力对象进行mask，注意力对象也是一个和潜变量图大小相同的，所以我们要扩大到原图大小）
        # if init_mask is None:
        #     init_mask = torch.nn.functional.interpolate((before_true_label_attention_map.detach().clone().mean(-1) / 
        #                                                  before_true_label_attention_map.detach().clone().mean(-1)
        #                                                  .max()).unsqueeze(0).unsqueeze(0),
        #                                                 init_image.shape[-2:], mode="bilinear").clamp(0, 1)
        #     if hard_mask:
        #         init_mask = init_mask.gt(0.5).float() # 大于0.5都为1，小于0.5都是0


        # 16. 对获取到的图像进行逆处理得到最终的图像
        init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                1 - init_mask) * init_image # 噪声潜变量 加上 原始图像（要用mask区分区域）
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        # 17. 使用target模型对生成的图像进行预测，并将其和真实标签做交叉熵，拉大差距
        pred = classifier(out_image)

        # 18. 损失函数构成
        # 攻击损失1，让预测结果不准
        attack_loss = - cross_entro(pred, label) * args.attack_loss_weight
        # “Deceive” Strong Diffusion Model. Details please refer to Section 3.3 
        # 攻击损失2，让注意力图像的方差变大，让注意力关注全局（本来只关注text对应的部分）
        variance_cross_attn_loss = after_true_label_attention_map.mean() * args.cross_attn_loss_weight * 0.1
        # Preserve Content Structure. Details please refer to Section 3.4
        # 攻击损失3，控制语义损失不要过大（让加了扰动和没加扰动的图像的自注意力图拉近）
        self_attn_loss = controller.loss * args.self_attn_loss_weight
        loss = self_attn_loss + attack_loss + variance_cross_attn_loss
        # loss = self_attn_loss + variance_cross_attn_loss

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"variance_cross_attn_loss: {variance_cross_attn_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 优化latent

    with torch.no_grad():
        controller.loss = 0
        controller.reset()

        latents = torch.cat([original_latent, latent]) # 原始 latent 和攻击优化后的 latent

        # 通过优化后的 latent 生成图片 其中 original_latent 的加入是一个技巧
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    # 19. 生成最终的图像，并进行处理
    out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
            1 - init_mask) * init_image # 图像解码
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    # 20. 得到最终图像的准确度
    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))


    logit = torch.nn.Softmax()(pred) # 置信度
    print("after_pred:", pred_label, logit[0, pred_label[0]]) # 预测的标签以及其置信度
    print("after_true:", label, logit[0, label[0]])           # 真实的标签以及其置信度

    """
            ==========================================
            ============= Visualization ==============
            ==========================================
    """

    # 21. 不加入mask的图像
    image = latent2image(model.vae, latents.detach()) # image from latents（simple of the up code）

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() # real iamge
    perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
            1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real # 要通过mask只保留某个区域的扰动
    image = (perturbed * 255).astype(np.uint8)
    view_images(np.concatenate([real, perturbed]) * 255, show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
                                                                     "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print("L1: {}\tL2: {}\tLinf: {}".format(L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)))

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    reset_attention_control(model)

    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=7, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionBefore.jpg".format(save_path))
    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=7, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionAfter.jpg".format(save_path), select=1)
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionBefore.jpg".format(save_path))
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionAfter.jpg".format(save_path), select=1)

    return image[0], 0, 0
