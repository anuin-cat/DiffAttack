from typing import Union, Tuple
import torch
import abc


class AttentionControl(abc.ABC):
    # abc：python中的抽象库类，用于定义抽象类组件

    def __init__(self):
        self.cur_step = 0 # 当前步数
        self.num_att_layers = -1 # 注意力层数
        self.cur_att_layer = 0 # 当前注意力所在层

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # __call__ 方法放在类中时，代表类实例化的对象 fun 直接调用时使用的方法
        # 例如 fun = AttentionControl()
        #      fun() 就会调用__call__方法
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1 # 当前关注层递增
        if self.cur_att_layer == self.num_att_layers: # 如果已经到了最后一层
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # 对这个位置对应的 key 拼接上一个当前的 attention （每个key存有多个 attention ）
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 14 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        # 在 attention_store 每个item 加上对应的 step_store 的 item
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # 将 attention_store 中每个 item 都除以 cur_step 得到 average_attention
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        # 将所有存储的内容都置空
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    # Unet 中所有 CrossAttention 操作中的forward都会调用这里边的forward函数 
    # 具体可查看 diff_latent_attack 中的 register_attention_control 函数
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps # 变成一个元组，如果srs=0.5，那么 srs = 0,srs = (0.0, 0.5)
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1]) # 得到一个范围
        self.loss = 0
        self.criterion = torch.nn.MSELoss()

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # 原图的注意力 + 修改后的图的注意力（自注意力和交叉注意力共享一个变量）
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                """
                        ==========================================
                        ========= Self Attention Control =========
                        === Details please refer to Section 3.4 ==
                        ==========================================

                        对自注意力进行控制，使其生成结果不会很违和（拉近原图和修改后的图的自注意力）
                """
                self.loss += self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
