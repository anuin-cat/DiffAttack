import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
import torchvision.transforms as transforms
import torch

# 导入dataloader
from torch.utils.data import DataLoader

from prettytable import PrettyTable
from art.estimators.classification import PyTorchClassifier
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
import warnings
import pytorch_fid.fid_score as fid_score

warnings.filterwarnings("ignore")


def model_selection(name):
    if name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception":
        model = models.inception_v3(pretrained=True)
    elif name == "deit-b":
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=True
        )
    elif name == "deit-s":
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True
        )
    elif name == "mixer-b":
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=True
        )
    elif name == "mixer-l":
        model = timm.create_model(
            'mixer_l16_224',
            pretrained=True
        )
    elif name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()


def model_transfer(clean_img, adv_img, label, res, save_path=r"/home/DiffAttack/output", fid_path=None):
    log = open(os.path.join(save_path, "log.txt"), mode="w", encoding="utf-8")
    # models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin", 'deit-b', 'deit-s',
    #                         'mixer-b', 'mixer-l', 'tf2torch_adv_inception_v3', 'tf2torch_ens3_adv_inc_v3',
    #                         'tf2torch_ens4_adv_inc_v3', 'tf2torch_ens_adv_inc_res_v2']
    models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin", 'deit-b', 'deit-s',
                            'mixer-b', 'mixer-l']
    # models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin"]
    all_clean_accuracy = []
    all_adv_accuracy = []
    table = PrettyTable(["model", "acc org", "acc adv", "avg org", "avg adv"])

    for name in models_transfer_name:
        print("\n*********Transfer to {}********".format(name))
        print("\n*********Transfer to {}********".format(name), file=log)
        model = model_selection(name)
        model.eval()
        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=1000,
            preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            device_type='gpu',
        )

        clean_pred = f_model.predict(clean_img, batch_size=50)

        accuracy = np.sum((np.argmax(clean_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(clean_pred, axis=1) == label) / len(label)
        print("Accuracy on benign examples: {}%".format(accuracy * 100))
        print("Accuracy on benign examples: {}%".format(accuracy * 100), file=log)
        all_clean_accuracy.append(accuracy * 100)

        adv_pred = f_model.predict(adv_img, batch_size=50)
        accuracy = np.sum((np.argmax(adv_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(adv_pred, axis=1) == label) / len(label)
        print("Accuracy on adversarial examples: {}%".format(accuracy * 100))
        print("Accuracy on adversarial examples: {}%".format(accuracy * 100), file=log)
        all_adv_accuracy.append(accuracy * 100)

        table.add_row([name, "{:.2f}%".format(all_clean_accuracy[-1]), "{:.2f}%".format(all_adv_accuracy[-1]),
                          "{:.2f}%".format(np.mean(all_clean_accuracy)), "{:.2f}%".format(np.mean(all_adv_accuracy))])

    print(table)
    print(table, file=log)
    # print("clean_accuracy: ", "\t".join([str(x) for x in all_clean_accuracy]), file=log)
    # print("adv_accuracy: ", "\t".join([str(x) for x in all_adv_accuracy]), file=log)

    # fid = fid_score.main(save_path if fid_path is None else fid_path)
    # print("\n*********fid: {}********".format(fid))
    # print("\n*********fid: {}********".format(fid), file=log)

    log.close()




def model_transfer_full(res, dataset, adv_dataset, save_path=r"/home/DiffAttack/output_full", fid_path=None):

    # 1. 参数设置
    log = open(os.path.join(save_path, "log.txt"), mode="w", encoding="utf-8")
    models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", 
                            "vit", "swin", 'deit-b', 'deit-s','mixer-b', 'mixer-l']
    
    all_clean_accuracy = []
    all_adv_accuracy = []
    table = PrettyTable(["model", "acc org", "acc adv", "avg org", "avg adv"])

    transfers = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    adv_dataset.transforms = transfers
    dataset.transforms = transfers

    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=10)
    adv_dataloader = DataLoader(adv_dataset, batch_size=50, shuffle=False, num_workers=10)


    # 2. 使用不同的模型进行测试
    for name in models_transfer_name:
        print("\n*********Transfer to {}********".format(name))
        accuracy = 0
        adv_accuracy = 0

        for image, imageId, label_index, label in dataloader:
            model = model_selection(name)
            model.eval()
            pred = model(image.cuda()).cpu().detach().numpy()
            accuracy += np.sum(np.argmax(pred, axis=1) == label_index.numpy())

        for image, imageId, label_index, label in adv_dataloader:
            model = model_selection(name)
            model.eval()
            pred = model(image.cuda()).cpu().detach().numpy()
            adv_accuracy += np.sum(np.argmax(pred, axis=1) == label_index.numpy())

        accuracy = accuracy / len(dataset) * 100
        adv_accuracy = adv_accuracy / len(adv_dataset) * 100
        print("Accuracy on benign examples: {:.2f}%".format(accuracy))
        print("Accuracy on adversarial examples: {:.2f}%".format(adv_accuracy))

        all_clean_accuracy.append(accuracy)
        all_adv_accuracy.append(adv_accuracy)

        table.add_row([name, "{:.2f}%".format(accuracy), "{:.2f}%".format(adv_accuracy),
                       "{:.2f}%".format(np.mean(all_clean_accuracy)), "{:.2f}%".format(np.mean(all_adv_accuracy))])

    print(table)
    print(table, file=log)

    log.close()