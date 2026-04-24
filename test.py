# 导入PyTorch核心库
import torch
# 导入PyTorch的函数接口（如softmax, kl_div）
import torch.nn.functional as F
# 导入神经网络层、优化器相关模块
from torch import nn, optim

def train_kd_one_epoch(teacher, student, loader, optimizer, device, T=2.0, alpha=0.25):
    """
    知识蒸馏 训练单个epoch
    teacher: 训练好的大模型（教师模型，固定不动）
    student: 待训练的小模型（学生模型）
    loader: 训练数据加载器
    optimizer: 学生模型的优化器
    device: 运行设备（cpu / cuda）
    T: 蒸馏温度，越大输出越平滑，越能学到暗知识
    alpha: 蒸馏损失权重；(1-alpha) 是真实标签交叉熵损失权重
    """

    # -------------------------- 关键：模型模式切换 --------------------------
    # 教师模型设置为评估模式（关闭dropout/bn，输出稳定，不更新参数）
    teacher.eval()
    # 学生模型设置为训练模式（开启dropout/bn，正常训练更新参数）
    student.train()

    # 定义标准交叉熵损失函数（用于学生模型拟合真实标签）
    ce_loss_fn = nn.CrossEntropyLoss()

    # 记录一个epoch的总损失
    total_loss = 0.0

    # 遍历训练集的每一个batch数据（x=输入数据，y=真实标签）
    for x, y in loader:
        # 将数据和标签移动到指定设备（GPU/CPU）
        x, y = x.to(device), y.to(device)

        # 优化器梯度清零（避免上一步的梯度累积）
        optimizer.zero_grad()

        # -------------------------- 教师模型前向传播 --------------------------
        # torch.no_grad()：关闭梯度计算，节省显存+加速
        # 教师模型已经训练好，不需要计算梯度，也不更新参数
        with torch.no_grad():
            # 教师模型对输入x做前向推理，输出原始预测分数（logits）
            teacher_logits = teacher(x)

        # -------------------------- 学生模型前向传播 --------------------------
        # 学生模型对输入x做前向推理，输出原始预测分数（logits）
        student_logits = student(x)

        # -------------------------- 计算知识蒸馏损失 --------------------------
        # 1. 对教师logits做温度软化，得到概率分布
        p_t = F.softmax(teacher_logits / T, dim=-1)
        # 2. 对学生logits做温度软化，取对数概率（KL散度要求输入log概率）
        log_p_s = F.log_softmax(student_logits / T, dim=-1)
        # 3. 计算KL散度：让学生拟合教师的输出分布
        # 乘以 T*T 是为了抵消温度对梯度大小的影响（标准蒸馏做法）
        kd_loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

        # -------------------------- 计算真实标签损失 --------------------------
        # 标准交叉熵：让学生拟合真实标签y
        ce_loss = ce_loss_fn(student_logits, y)

        # -------------------------- 总损失 = 蒸馏损失 + 真实标签损失 --------------------------
        # alpha 控制教师知识权重，(1-alpha) 控制真实标签权重
        loss = alpha * kd_loss + (1 - alpha) * ce_loss

        # 反向传播：计算梯度
        loss.backward()
        # 优化器更新学生模型参数
        optimizer.step()

        # 累加当前batch的损失到总损失
        total_loss += loss.item()

    # 返回一个epoch的平均损失（总损失 / batch数量）
    return total_loss / len(loader)