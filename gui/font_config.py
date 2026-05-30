"""
matplotlib 中文字体配置
Windows 系统下使用微软雅黑或宋体
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform

def configure_chinese_font():
    """
    配置 matplotlib 中文字体。
    返回配置好的字体名称。
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows 候选字体（按优先级）
        font_candidates = [
            "Microsoft YaHei",      # 微软雅黑
            "SimHei",               # 黑体
            "SimSun",               # 宋体
            "KaiTi",                # 楷体
            "FangSong",             # 仿宋
            "Arial",                # 回退到英文
        ]
    else:
        # Linux/Mac 候选字体
        font_candidates = [
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Noto Sans CJK SC",
            "SimHei",
            "DejaVu Sans",
        ]
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 选择第一个可用字体
    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break
    
    if selected_font is None:
        # 没有任何中文字体，使用默认字体并发出警告
        print("⚠ 警告: 未找到中文字体，图表中文可能显示为方框")
        print("  可用字体:", sorted(available_fonts)[:20])
        selected_font = "Arial"
    
    # 全局配置
    matplotlib.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    print(f"✓ matplotlib 字体配置: {selected_font}")
    return selected_font