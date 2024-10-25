import re
from collections import defaultdict
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from typing import Optional, cast

def parse_log_file(file_path):
    # 正则表达式匹配日志中的 epoch 和 loss
    pattern = re.compile(r'\[Rank (\d+) \| DistributedDataParallel epoch (\d+)\].*? loss (\d+\.\d+)')
    
    # 用于存储每个 epoch 的 loss 值
    epoch_losses = defaultdict(lambda: dict())
    
    # 读取文件并解析
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                rank = int(match.group(1))
                epoch = int(match.group(2))
                loss = float(match.group(3))
                epoch_losses[epoch][rank] = loss
    
    # 计算每个 epoch 不同 rank 的平均 loss
    avg_epoch_losses = []
    for epoch, epoch_loss in epoch_losses.items():
        avg_epoch_losses.append(
            sum(epoch_loss.values()) / len(epoch_loss)
        )
    return avg_epoch_losses

def draw_compare_fig(w_col_loss: list[float], wo_col_loss: list[float], max_epoch: Optional[int]):
    epoch = len(w_col_loss)
    if max_epoch is not None:
        epoch = min(epoch, max_epoch)
    if wo_col_loss is not None:
        epoch = min(epoch, len(wo_col_loss))
    w_col_loss = w_col_loss[:epoch]
    wo_col_loss = wo_col_loss[:epoch]
    
    epochs = list(range(1, epoch + 1))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, w_col_loss, label='w/ COL', color='blue', marker='o')
    plt.plot(epochs, wo_col_loss, label='w/o COL', color='red', marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Loss with and without Col')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_compare.png')

def main():
    parser = ArgumentParser()
    parser.add_argument('--wo-col', type=str, default=None)
    parser.add_argument('--w-col', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=None)

    args = parser.parse_args()
    wo_col = cast(str, args.wo_col)
    wo_col_loss = parse_log_file(wo_col) 
    w_col = cast(Optional[str], args.w_col)
    w_col_loss = parse_log_file(w_col) if w_col is not None else None
    epoch = cast(Optional[int], args.epoch)
    
    draw_compare_fig(w_col_loss, wo_col_loss, epoch)

if __name__ == "__main__":
    main()
