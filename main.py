import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attentionControl import AttentionControlEdit
import diff_latent_attack
from PIL import Image
import numpy as np
import os
import glob
from other_attacks import model_transfer
import random
import sys
from natsort import ns, natsorted
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="/home/DiffAttack/output", type=str,
                    help='Where to save the adversarial examples, and other results')
parser.add_argument('--images_root', default=r"/home/DiffAttack/demo/images", type=str,
                    help='The clean images root directory')
parser.add_argument('--label_path', default=r"/home/DiffAttack/demo/labels.txt", type=str,
                    help='The clean images labels.txt')
parser.add_argument('--is_test', default=False, type=bool,
                    help='Whether to test the robustness of the generated adversarial examples')
parser.add_argument('--pretrained_diffusion_path',
                    default=r"stabilityai/stable-diffusion-2-base",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')

parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step', default=15, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations', default=30, type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name', default="resnet", type=str,
                    help='The surrogate model from which the adversarial examples are crafted')
parser.add_argument('--is_apply_mask', default=False, type=bool,
                    help='Whether to leverage pseudo mask for better imperceptibility (See Appendix D)')
parser.add_argument('--is_hard_mask', default=False, type=bool,
                    help='Which type of mask to leverage (See Appendix D)')

parser.add_argument('--guidance', default=2.5, type=float, help='guidance scale of diffusion models')
parser.add_argument('--attack_loss_weight', default=10, type=int, help='attack loss weight factor')
parser.add_argument('--cross_attn_loss_weight', default=10000, type=int, help='cross attention loss weight factor')
parser.add_argument('--self_attn_loss_weight', default=100, type=int, help='self attention loss weight factor')


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
    args = parser.parse_args()
    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name  # The surrogate model from which the adversarial examples are crafted.
    save_dir = args.save_dir  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)

    "If you set 'is_test' to True, please turn 'images_root' to the path of the output results' path."
    images_root = args.images_root  # The clean images' root directory.
    label_path = args.label_path  # The clean images' labels.txt.
    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)

    is_test = args.is_test  # Whether to test the robustness of the generated adversarial examples.

    print("\n******Attack based on Diffusion*********")

    # Change the path to "stabilityai/stable-diffusion-2-base" if you want to use the pretrained model.
    pretrained_diffusion_path = args.pretrained_diffusion_path

    # 1. 加载默认的 stable 扩撒模型
    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to('cuda:0')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    
    # 2. 加载图像
    "Attack a subset images"
    all_images = glob.glob(os.path.join(images_root, "*")) # 获取目录下所有路径并按照较好的规则排序
    all_images = natsorted(all_images, alg=ns.PATH)

    adv_images = []
    images = []
    clean_all_acc = 0
    adv_all_acc = 0

    if is_test:
        all_clean_images = glob.glob(os.path.join(save_dir, "*originImage*"))
        all_adv_images = glob.glob(os.path.join(save_dir, "*adv_image*"))
        for image_path, adv_image_path in zip(all_clean_images, all_adv_images):
            tmp_image = Image.open(image_path).convert('RGB')
            tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
            tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
            tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
            images.append(tmp_image)

            tmp_image = Image.open(adv_image_path).convert('RGB')
            tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
            tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
            tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
            adv_images.append(tmp_image)

        images = np.concatenate(images)
        adv_images = np.concatenate(adv_images)

        """
                Test the robustness of the generated adversarial examples across a variety of normally trained models or
                adversarially trained models.
        """
        # model_transfer(images, adv_images, label, res, save_path=save_dir, fid_path=images_root)
        model_transfer(images, adv_images, label, res, save_path=save_dir)

        sys.exit()

    # 3. 迭代所有图像
    for ind, image_path in enumerate(all_images):
        # 3.1 获取原始图像
        tmp_image = Image.open(image_path).convert('RGB') # .rjust(4, '0') 用于右对齐，左侧填充0
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, '0') + "_originImage.png"))

        # 3.2 获取对抗图像、攻击成功率
        adv_image, clean_acc, adv_acc = run_diffusion_attack(tmp_image, label[ind:ind + 1],
                                                             ldm_stable,
                                                             diffusion_steps, guidance=guidance,
                                                             res=res, model_name=model_name,
                                                             start_step=start_step,
                                                             iterations=iterations,
                                                             save_dir=os.path.join(save_dir,
                                                                                   str(ind).rjust(4, '0')), args=args)

        # 3.3 图像格式转换
        adv_image = adv_image.astype(np.float32) / 255.0
        adv_images.append(adv_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)

        clean_all_acc += clean_acc
        adv_all_acc += adv_acc

    print("Clean acc: {}%".format(clean_all_acc / len(all_images) * 100))
    print("Adv acc: {}%".format(adv_all_acc / len(all_images) * 100))

    images = np.concatenate(images)
    adv_images = np.concatenate(adv_images)

    """
            Test the robustness of the generated adversarial examples across a variety of normally trained models or
            adversarially trained models.
    """
    model_transfer(images, adv_images, label, res, save_path=save_dir)
