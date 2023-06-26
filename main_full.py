import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_latent_attack
from PIL import Image
import numpy as np
import os
import glob
from other_attacks import model_transfer, model_transfer_full
import random
import sys
from natsort import ns, natsorted
import argparse 
from utils import ImageNet, AdvImageNet
from torchvision.transforms import transforms

'''
    全量攻击，这里和main的主要区别在数据集上，main数据集只是一个简单的demo版本
    这里使用1000张图片进行攻击
'''

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="/home/DiffAttack/result/", type=str)
parser.add_argument('--images_root', default=r"/home/DiffAttack/dataset/images", type=str)
parser.add_argument('--csv_path', default=r"/home/DiffAttack/dataset/dev_dataset_20.csv", type=str)
parser.add_argument('--is_test', default=False, type=bool, help='Whether just to test')
parser.add_argument('--pretrained_diffusion_path', default=r"stabilityai/stable-diffusion-2-base", type=str)

parser.add_argument('--diffusion_steps', default=30, type=int, help='Total DDIM sampling steps') # 20
parser.add_argument('--start_step', default=25, type=int, help='Which DDIM step to start the attack') # 15
parser.add_argument('--iterations', default=30, type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name', default="resnet", type=str, help='The surrogate model')

parser.add_argument('--is_apply_mask', default=False, type=bool)
parser.add_argument('--is_hard_mask', default=False, type=bool)

parser.add_argument('--guidance', default=2.5, type=float)
parser.add_argument('--attack_loss_weight', default=10, type=int)
parser.add_argument('--cross_attn_loss_weight', default=10000, type=int)
parser.add_argument('--self_attn_loss_weight', default=100, type=int)

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(42)


def run_diffusion_attack(image, label, diffusion_model, diffusion_steps, guidance=2.5,
                         self_replace_steps=1., save_dir=r"/home/DiffAttack/output", res=224,
                         model_name="inception", start_step=15, iterations=30, args=None):
    # 1. 获取控制器
    controller = AttentionControlEdit(diffusion_steps, self_replace_steps)

    # 2. 获取对抗样本
    adv_image, clean_acc, adv_acc = diff_latent_attack.diffattack(diffusion_model, label, controller,
                                                                  num_inference_steps=diffusion_steps,
                                                                  guidance_scale=guidance,
                                                                  image=image,
                                                                  save_path=save_dir, res=res, model_name=model_name,
                                                                  start_step=start_step,
                                                                  iterations=iterations, args=args)
    return adv_image, clean_acc, adv_acc


if __name__ == "__main__":
    # 1. 参数设置
    print("\n******Attack based on Diffusion*********")
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.csv_path.split("_")[-1].split(".")[0]) + '/'

    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name  # The surrogate model from which the adversarial examples are crafted.
    save_dir = args.save_dir  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)

    # 2. 数据集 与 模型加载
    if not args.is_test:
        ldm_stable = StableDiffusionPipeline.from_pretrained(args.pretrained_diffusion_path).to('cuda:0')
        ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    dataset = ImageNet(args.images_root, args.csv_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    # 3. 迭代所有图像
    adv_images = []
    images = []
    clean_all_acc = 0
    adv_all_acc = 0
    for ind, (image, imageId, label_index, label) in enumerate(dataloader):
        if args.is_test:
            break
        # 3.1 类型转换
        imageId = imageId[0]
        label = label[0]
        label_index = label_index.numpy()
        image = transforms.ToPILImage()(image[0])
        image.save(os.path.join(save_dir, imageId + "_originImage.png"))

        # 3.2 获取对抗图像、攻击成功率
        adv_image, clean_acc, adv_acc = run_diffusion_attack(image, label_index,
                                                             ldm_stable,
                                                             diffusion_steps, guidance=guidance,
                                                             res=res, model_name=model_name,
                                                             start_step=start_step,
                                                             iterations=iterations,
                                                             save_dir=os.path.join(save_dir,
                                                                                   imageId), args=args)
        clean_all_acc += clean_acc
        adv_all_acc += adv_acc

    if not args.is_test:
        print("Clean acc: {}%".format(clean_all_acc / len(dataset) * 100))
        print("Adv acc: {}%".format(adv_all_acc / len(dataset) * 100))

    """
            Test the robustness of the generated adversarial examples across a variety of normally trained models or
            adversarially trained models.
    """
    adv_dataset = AdvImageNet(args.save_dir, args.csv_path)
    model_transfer_full(res, dataset, adv_dataset, save_path=save_dir)
