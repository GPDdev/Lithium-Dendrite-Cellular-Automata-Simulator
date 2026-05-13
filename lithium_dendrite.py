import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

# 尝试启用常见中文字体；没有也能正常运行
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LithiumDendriteCA:
    """
    基于元胞自动机(CA)的锂枝晶生长模拟
    0 = 电解液
    1 = 金属锂沉积
    """

    def __init__(
        self,
        width=120,
        height=120,
        steps=180,
        eta=0.62,
        D=0.28,
        sei_uniformity=0.72,
        anisotropy=0.35,
        noise=0.08,
        seed=42,
    ):
        self.width = width
        self.height = height
        self.steps = steps
        self.eta = eta
        self.D = D
        self.sei_uniformity = sei_uniformity
        self.anisotropy = anisotropy
        self.noise = noise
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.grid[0, :] = 1

        # 初始表面微扰，模拟粗糙电极/缺陷位点
        for x in range(0, width, 6):
            if self.rng.random() < 0.45:
                self.grid[1, x] = 1

        self.frames = [self.grid.copy()]
        self.records = []

    def get_interface_candidates(self):
        g = self.grid
        candidates = []
        H, W = g.shape
        for y in range(1, H):
            for x in range(W):
                if g[y, x] != 0:
                    continue
                y0, y1 = max(0, y - 1), min(H, y + 2)
                x0, x1 = max(0, x - 1), min(W, x + 2)
                if np.any(g[y0:y1, x0:x1] == 1):
                    candidates.append((y, x))
        return candidates

    def local_probability(self, y, x):
        g = self.grid
        H, W = g.shape

        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        neigh = g[y0:y1, x0:x1]
        n_li = np.sum(neigh == 1)

        support = 1.0 if y > 0 and g[y - 1, x] == 1 else 0.0

        lateral = 0
        if x > 0 and g[y, x - 1] == 1:
            lateral += 1
        if x < W - 1 and g[y, x + 1] == 1:
            lateral += 1
        tip_factor = support * (1.6 - 0.4 * lateral)

        filled = np.where(g[:, x] == 1)[0]
        col_top = filled.max() if len(filled) else 0
        front_bias = np.exp(-abs(y - (col_top + 1)) / 6.0)

        hotspot = 1.0 + (1.0 - self.sei_uniformity) * self.rng.uniform(0.0, 1.0)
        diffusion_smoothing = 0.65 + 0.7 * self.D

        j_proxy = (
            0.45 * self.eta
            + 0.15 * n_li / 8.0
            + 0.20 * support
            + self.anisotropy * 0.25 * tip_factor
            + 0.20 * front_bias
        ) * hotspot / diffusion_smoothing

        p = 0.02 + 0.42 * np.tanh(1.35 * j_proxy)
        p += 0.05 * self.D * min(n_li / 6.0, 1.0)
        p += self.rng.normal(0.0, self.noise * 0.03)

        return float(np.clip(p, 0.001, 0.92))

    def measure(self, step):
        g = self.grid
        heights = []
        for x in range(self.width):
            ys = np.where(g[:, x] == 1)[0]
            heights.append(ys.max() if len(ys) else 0)
        heights = np.array(heights)

        deposited_cells = int(np.sum(g == 1) - self.width)
        max_height = int(heights.max())
        mean_height = float(heights.mean())
        roughness = float(heights.std())

        tips = 0
        for x in range(self.width):
            ys = np.where(g[:, x] == 1)[0]
            if len(ys):
                top = ys.max()
                if top < self.height - 1 and g[top + 1, x] == 0:
                    tips += 1

        coverage = float(np.mean(heights > 0))

        self.records.append(
            {
                'step': step,
                'deposited_cells': deposited_cells,
                'max_height': max_height,
                'mean_height': mean_height,
                'roughness': roughness,
                'active_tips': int(tips),
                'coverage_ratio': coverage,
            }
        )

    def step_forward(self):
        candidates = self.get_interface_candidates()
        if not candidates:
            return False

        probs = np.array([self.local_probability(y, x) for y, x in candidates])
        expected_new = max(1, int(1 + self.eta * self.width * 0.055))
        chosen_count = min(len(candidates), expected_new)

        if probs.sum() <= 0:
            idx = self.rng.choice(len(candidates), size=chosen_count, replace=False)
        else:
            p = probs / probs.sum()
            idx = self.rng.choice(len(candidates), size=chosen_count, replace=False, p=p)

        for i in idx:
            y, x = candidates[i]
            self.grid[y, x] = 1

        return True

    def run(self):
        self.measure(0)
        for t in range(1, self.steps + 1):
            ok = self.step_forward()
            self.frames.append(self.grid.copy())
            self.measure(t)
            if not ok:
                break
        return pd.DataFrame(self.records)


def save_outputs(df, frames, outdir):
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, 'dendrite_growth_data.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frames[-1], origin='lower', interpolation='nearest', cmap='viridis')
    ax.set_title('锂枝晶最终形貌图')
    ax.set_xlabel('横向位置 x')
    ax.set_ylabel('纵向位置 y')
    fig.tight_layout()
    final_png = os.path.join(outdir, 'dendrite_final.png')
    fig.savefig(final_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['step'], df['max_height'], label='最大高度')
    ax.plot(df['step'], df['mean_height'], label='平均高度')
    ax.plot(df['step'], df['roughness'], label='粗糙度')
    ax.set_xlabel('演化步数')
    ax.set_ylabel('指标值')
    ax.set_title('锂枝晶生长指标随时间变化')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    metrics_png = os.path.join(outdir, 'growth_metrics.png')
    fig.savefig(metrics_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], origin='lower', interpolation='nearest', cmap='viridis', animated=True)
    ax.set_xlabel('横向位置 x')
    ax.set_ylabel('纵向位置 y')

    def update(i):
        im.set_array(frames[i])
        ax.set_title(f'锂枝晶生长过程（step={i}）')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
    gif_path = os.path.join(outdir, 'dendrite_growth.gif')
    try:
        ani.save(gif_path, writer='pillow', fps=12)
    except Exception as e:
        print('GIF 保存失败：', e)
        gif_path = ''
    plt.close(fig)

    key_ids = np.linspace(0, len(frames) - 1, 6, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, idx in zip(axes.flat, key_ids):
        ax.imshow(frames[idx], origin='lower', interpolation='nearest', cmap='viridis')
        ax.set_title(f'第 {idx} 步')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('锂枝晶生长关键阶段')
    fig.tight_layout()
    key_png = os.path.join(outdir, 'growth_stages.png')
    fig.savefig(key_png, dpi=180)
    plt.close(fig)

    summary_path = os.path.join(outdir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('基于元胞自动机的锂枝晶生长模拟结果\n')
        f.write('=' * 40 + '\n')
        f.write(f"总步数: {int(df['step'].max())}\n")
        f.write(f"最终沉积单元数: {int(df['deposited_cells'].iloc[-1])}\n")
        f.write(f"最终最大高度: {int(df['max_height'].iloc[-1])}\n")
        f.write(f"最终平均高度: {df['mean_height'].iloc[-1]:.3f}\n")
        f.write(f"最终粗糙度: {df['roughness'].iloc[-1]:.3f}\n")
        f.write(f"最终活性尖端数: {int(df['active_tips'].iloc[-1])}\n")

    return {
        'csv': csv_path,
        'final_png': final_png,
        'metrics_png': metrics_png,
        'gif': gif_path,
        'stages_png': key_png,
        'summary': summary_path,
    }


def main():
    model = LithiumDendriteCA(
        width=120,
        height=120,
        steps=180,
        eta=0.64,
        D=0.26,
        sei_uniformity=0.70,
        anisotropy=0.40,
        noise=0.08,
        seed=42,
    )
    df = model.run()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(base_dir, 'lithium_dendrite_outputs')
    paths = save_outputs(df, model.frames, outdir=outdir)

    print('\n模拟完成，输出文件如下：')
    for k, v in paths.items():
        print(f'{k}: {v}')
    print('\n末尾几行生长数据：')
    print(df.tail())


if __name__ == '__main__':
    main()
