import torch
import argparse
import sys
import os

# 尝试导入 safetensors（用于 .safetensors 文件）
try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

def load_state_dict(path):
    """支持 .pth 和 .safetensors 两种格式"""
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    
    ext = os.path.splitext(path)[1].lower()
    
    try:
        if ext == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                print("❌ 需要安装 safetensors 库来加载 .safetensors 文件")
                print("💡 请运行: pip install safetensors")
                sys.exit(1)
            print(f"📦 正在加载 safetensors 文件: {path}")
            state_dict = load_file(path, device='cpu')
        elif ext in [".pth", ".pt", ".bin"]:
            print(f"📦 正在加载 PyTorch 文件: {path}")
            # 优先尝试 weights_only=True（更安全），失败则回退
            try:
                state_dict = torch.load(path, map_location='cpu', weights_only=True)
            except (pickle.UnpicklingError, TypeError, RuntimeError):
                print("⚠️ weights_only=True 加载失败，尝试 weights_only=False...")
                state_dict = torch.load(path, map_location='cpu', weights_only=False)
        else:
            print(f"❌ 不支持的文件格式: {ext}")
            sys.exit(1)
        
        # 兼容：如果加载的是整个模型对象，提取 state_dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        
        return state_dict
    except Exception as e:
        print(f"❌ 加载失败: {type(e).__name__}: {e}")
        sys.exit(1)

def print_all_weights(state_dict, max_elements=100):
    """打印所有权重（大张量会截断显示）"""
    for name, param in state_dict.items():
        print(f"\n{'='*70}")
        print(f"🔖 参数名: {name}")
        print(f"📐 形状: {param.shape}")
        print(f"🔢 数据类型: {param.dtype}")
        print(f"📊 元素总数: {param.numel():,}")
        
        # 避免打印超大张量
        if param.numel() <= max_elements:
            print(f"📋 值:\n{param}")
        else:
            flat = param.flatten()
            print(f"📋 前{max_elements//2}个值: {flat[:max_elements//2].tolist()}")
            print(f"📋 后{max_elements//2}个值: {flat[-max_elements//2:].tolist()}")
            print(f"⚠️  共 {param.numel()} 个元素，已截断显示")
        print(f"{'='*70}")

def print_summary(state_dict):
    print(f"\n📦 模型包含 {len(state_dict)} 个参数\n{'-'*70}")
    for name, param in state_dict.items():
        print(f"{name:60} | {str(param.shape):25} | {param.dtype}")
    print(f"{'-'*70}")

def print_statistics(state_dict):
    print(f"\n📊 权重统计信息\n{'-'*80}")
    total_params = 0
    for name, param in state_dict.items():
        num_params = param.numel()
        total_params += num_params
        param_flat = param.float().flatten()
        print(f"{name:55} | 数量: {num_params:>12,} | 均值: {param_flat.mean().item():>9.4f} | 标准差: {param_flat.std().item():>9.4f}")
    print(f"{'-'*80}\n✨ 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

def print_partial_weights(state_dict, top_n=5, show_elements=10):
    print(f"\n🔍 展示前 {top_n} 个参数的部分值（每个显示前{show_elements}个元素）\n")
    for i, (name, param) in enumerate(state_dict.items()):
        if i >= top_n:
            break
        print(f"\n{name}")
        print(f"形状: {param.shape}, 类型: {param.dtype}, 元素数: {param.numel():,}")
        flat = param.flatten()[:show_elements]
        print(f"前{show_elements}个值: {flat.tolist()}")

def main():
    parser = argparse.ArgumentParser(description="打印 .pth / .safetensors 文件中的权重信息")
    parser.add_argument("file", help="权重文件路径 (.pth, .pt, .bin, .safetensors)")
    parser.add_argument("--mode", choices=["all", "summary", "stats", "partial"], default="summary",
                        help="打印模式：all（全部）, summary（摘要）, stats（统计）, partial（部分）")
    parser.add_argument("--top", type=int, default=5, help="partial 模式下显示的参数个数")
    parser.add_argument("--max-elements", type=int, default=100, help="all 模式下单个张量最大显示元素数")
    args = parser.parse_args()

    state_dict = load_state_dict(args.file)

    if args.mode == "all":
        print_all_weights(state_dict, max_elements=args.max_elements)
    elif args.mode == "summary":
        print_summary(state_dict)
    elif args.mode == "stats":
        print_statistics(state_dict)
    elif args.mode == "partial":
        print_partial_weights(state_dict, top_n=args.top)

if __name__ == "__main__":
    # 忽略 pickle 安全警告（仅在确认文件可信时）
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()