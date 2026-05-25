"""
行星磁场标度律 B ∝ M·ω 对数空间检验（终极修正版）
核心：log10(B) = a * log10(M·ω) + b
使用 scipy.stats.linregress 执行标准线性回归
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("行星磁场标度律：对数空间标准线性回归（终极版）")
print("=" * 60)

# =============================================
# 1. 数据（NASA Planetary Fact Sheet）
# =============================================
planets = {
    "水星":   {"M": 0.0553, "P": 58.646,  "B": 300},
    "金星":   {"M": 0.815,  "P": 243.02,  "B": 0.5},
    "地球":   {"M": 1.000,  "P": 0.9973,  "B": 31000},
    "火星":   {"M": 0.107,  "P": 1.026,   "B": 0.01},
    "木星":   {"M": 317.8,  "P": 0.4135,  "B": 420000},
    "土星":   {"M": 95.16,  "P": 0.4440,  "B": 22000},
    "天王星": {"M": 14.54,  "P": 0.7183,  "B": 23000},
    "海王星": {"M": 17.15,  "P": 0.6713,  "B": 14000},
}

names = list(planets.keys())
M_all = np.array([planets[n]["M"] for n in names])
P_all = np.array([planets[n]["P"] for n in names])
B_all = np.array([planets[n]["B"] for n in names])
omega_all = 2 * np.pi / P_all
Momega_all = M_all * omega_all

# 活跃发电机行星索引
active_idx = [0, 2, 4, 5, 6, 7]
active_names = [names[i] for i in active_idx]
M_act = M_all[active_idx]
omega_act = omega_all[active_idx]
B_act = B_all[active_idx]
Momega_act = Momega_all[active_idx]

# =============================================
# 2. 对数空间线性回归（终极修正）
# =============================================
log_Momega = np.log10(Momega_act)
log_B = np.log10(B_act)

slope, intercept, r_value, p_value, std_err = stats.linregress(log_Momega, log_B)
r_squared = r_value ** 2
deviation = abs(slope - 1.0) / 1.0 * 100

print(f"\n对数空间线性回归结果：")
print(f"  log10(B) = {slope:.4f} × log10(M·ω) + {intercept:.4f}")
print(f"  R² = {r_squared:.4f}")
print(f"  p值 = {p_value:.6e}")
print(f"  斜率95%置信区间：[{slope-2*std_err:.4f}, {slope+2*std_err:.4f}]")
print(f"  与理论值1.00的偏差：{deviation:.2f}%")

if r_squared > 0.99:
    print(f"  ✅ 拟合极好")
elif r_squared > 0.95:
    print(f"  ✅ 拟合优秀")
elif r_squared > 0.9:
    print(f"  ⚠️ 拟合良好")
else:
    print(f"  ❌ 拟合一般")

# =============================================
# 3. Jackknife稳健性检验
# =============================================
n = len(active_names)
jk_results = []
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    s_jk, inc_jk, r_jk, p_jk, _ = stats.linregress(log_Momega[mask], log_B[mask])
    pred_B = 10 ** (s_jk * log_Momega[i] + inc_jk)
    err = abs(pred_B - B_act[i]) / B_act[i] * 100
    jk_results.append({"name": active_names[i], "slope": s_jk, "pred_B": pred_B, "err": err})

print(f"\nJackknife检验：")
for r in jk_results:
    print(f"  {r['name']}: 实际={B_act[active_names.index(r['name'])]:.0f} nT, 预测={r['pred_B']:.0f} nT, 误差={r['err']:.1f}%")

# =============================================
# 4. 可视化
# =============================================
fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(log_Momega, log_B, s=120, c='#2196F3', edgecolors='darkblue', linewidths=1.5, zorder=5)
for i, name in enumerate(active_names):
    ax.annotate(name, (log_Momega[i], log_B[i]), textcoords="offset points", xytext=(8, 8), fontsize=11)

x_range = np.linspace(log_Momega.min()-0.5, log_Momega.max()+0.5, 100)
ax.plot(x_range, slope*x_range + intercept, 'r-', linewidth=2, label=f'拟合线 (斜率={slope:.2f}, R²={r_squared:.4f})')
ax.plot(x_range, 1.0*x_range + intercept + (1.0-slope)*log_Momega.mean(), 'g--', linewidth=1.5, alpha=0.7, label='理论线 (斜率=1.00)')

# 失败案例
ax.scatter(np.log10(Momega_all[1]), np.log10(B_all[1]), s=80, c='gray', marker='x', linewidths=2)
ax.scatter(np.log10(Momega_all[3]), np.log10(B_all[3]), s=80, c='gray', marker='x', linewidths=2)
ax.annotate('金星', (np.log10(Momega_all[1]), np.log10(B_all[1])), textcoords="offset points", xytext=(-10, -20), fontsize=9, color='gray')
ax.annotate('火星', (np.log10(Momega_all[3]), np.log10(B_all[3])), textcoords="offset points", xytext=(8, -20), fontsize=9, color='gray')

ax.set_xlabel('log10(M·ω)', fontsize=13)
ax.set_ylabel('log10(B)', fontsize=13)
ax.set_title('行星磁场标度律 B ∝ M·ω', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('行星磁场_终极修正.png', dpi=200)
print(f"\n图表已保存：行星磁场_终极修正.png")

print(f"\n检验完成。如果R²>0.99且斜率≈1.0，则B∝M·ω获得极强支持。")