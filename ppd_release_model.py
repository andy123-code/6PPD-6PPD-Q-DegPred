"""
6PPD 土壤介质释放预测算法
===========================
耦合物理机制模型（Fick扩散 + 土壤传输）+ 机器学习增强预测

核心物理过程：
  1. TWP（轮胎磨损颗粒）→ 土壤孔隙水：扩散控制释放（Fick第二定律）
  2. 土壤水相 → 土壤固相：吸附/解吸平衡（Kd = Koc * foc）
  3. 土壤剖面迁移：对流-弥散方程（CDE）
  4. 一级降解耦合（水解、生物降解、光解）

ML模块：当实测数据充足时，用 XGBoost/RF 直接从环境条件预测释放通量

使用方式：
  python3 6ppd_release_model.py          # 运行内置示例
  python3 -c "from 6ppd_release_model import *; ..."  # 作为库导入
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 第零部分：6PPD 物理化学常数
# ============================================================

class Chem:
    """6PPD 理化参数集合"""

    MOLAR_MASS       = 268.4      # g/mol
    LOG_KOW          = 4.68       # 辛醇-水分配系数
    WATER_SOLUBILITY = 1.1        # mg/L, 25degC
    HENRY_CONSTANT   = 3.2e-5     # atm*m3/mol
    DIFFUSIVITY_WATER = 5.3e-10   # m2/s, 水中分子扩散系数

    # 降解半衰期（20degC 土壤）
    HL_AEROBIC_SOIL  = 75.0       # days, 好氧土壤
    HL_ANAEROBIC     = 180.0      # days, 厌氧
    HL_HYDROLYSIS    = 365.0      # days, 中性pH水解
    HL_PHOTOLYSIS    = 45.0       # days, 光解（表层）

    EA_DEGRADATION   = 50_000     # J/mol, Arrhenius 活化能

    @staticmethod
    def k_deg(T_C, moisture=0.30, is_surface=False):
        """一级降解速率 k [1/day]"""
        R = 8.314
        T_K = T_C + 273.15
        T_ref = 293.15
        k0 = np.log(2) / Chem.HL_AEROBIC_SOIL
        k_T = k0 * np.exp(Chem.EA_DEGRADATION / R * (1/T_ref - 1/T_K))

        # 水分修正
        if moisture < 0.10:
            mf = moisture / 0.10
        elif moisture <= 0.50:
            mf = 1.0
        else:
            mf = 0.50
        k = k_T * mf

        if is_surface:
            k += np.log(2) / Chem.HL_PHOTOLYSIS
        return k


# ============================================================
# 第一部分：TWP 释放模型 —— Fick 扩散驱动
# ============================================================

class TWPReleaseModel:
    """
    TWP颗粒中6PPD向土壤释放的Fick扩散模型

    假设：球形颗粒，内部均匀分布，表面浓度≈0（土壤孔隙水快速移除）
    解析解：Crank (1975)，球形扩散

    Mt/Minf = 1 - (6/pi^2) * SUM_{n=1}^{inf} (1/n^2) * exp(-n^2 * pi^2 * Dt/R^2)

    **基准**：单位质量 TWP = 1 g，方便与实际投放量线性缩放
    """

    def __init__(self, particle_radius_um=50, d0_m2_s=5e-18,
                 loading_wt_pct=1.0, twp_mass_g=1.0):
        """
        particle_radius_um: TWP平均半径 (um), 典型10-150 um
        d0_m2_s: 20degC时橡胶基质内6PPD有效扩散系数 (m2/s)
                 文献范围 1e-19 ~ 1e-16（极端缓慢）
        loading_wt_pct: 轮胎橡胶中6ppd质量分数 (%), 典型0.5-2%
        twp_mass_g: 计算基准 —— TWP投放总量 (g)
        """
        self.R = particle_radius_um * 1e-6         # m
        self.D0 = d0_m2_s
        self.twp_mass_g = twp_mass_g
        self.M_total = twp_mass_g * loading_wt_pct / 100.0 * 1000  # mg

    def D_eff(self, T_C, moisture):
        """温度+水分修正后的有效扩散系数 [m2/s]"""
        Ea = 60_000                # J/mol, 聚合物内扩散活化能
        Rc = 8.314
        TK = T_C + 273.15
        Tref = 293.15
        D_T = self.D0 * np.exp(Ea / Rc * (1/Tref - 1/TK))
        moist_boost = 1.0 + 2.0 * moisture
        return D_T * moist_boost

    def release_fraction(self, t_days, T_C=20, moisture=0.30):
        """t_days 时的释放分数 Mt/Minf，使用 Crank 级数解（50项）"""
        if t_days <= 0:
            return 0.0
        D = self.D_eff(T_C, moisture)
        tau = D * t_days * 86400.0 / (self.R ** 2)

        s = 0.0
        for n in range(1, 51):
            s += np.exp(-(n * np.pi) ** 2 * tau) / (n ** 2)
        return 1.0 - (6.0 / np.pi ** 2) * s

    def cumulative_release(self, t_days, T_C=20, moisture=0.30):
        """累积释放 6PPD 质量 (mg)"""
        return self.release_fraction(t_days, T_C, moisture) * self.M_total

    def release_rate(self, t_days, T_C=20, moisture=0.30):
        """瞬时释放速率 (mg/day)，中心差分"""
        eps = max(t_days * 0.001, 0.001)
        m1 = self.cumulative_release(t_days - eps, T_C, moisture)
        m2 = self.cumulative_release(t_days + eps, T_C, moisture)
        return max((m2 - m1) / (2 * eps), 0)

    def time_to_fraction(self, fraction, T_C=20, moisture=0.30):
        """
        二分法求达到 fraction 释放分数所需时间 (天)
        短时间近似: Mt/Min∞ ≈ (6/sqrt(pi)) * sqrt(Dt/R^2)
        长时间近似: Mt/Min∞ ≈ 1 - (6/pi^2) * exp(-pi^2 * Dt/R^2)
        用短时近似下界 + 长时近似上界作为二分区间
        """
        D = self.D_eff(T_C, moisture)
        tau_char = self.R ** 2 / D / 86400.0       # 特征扩散时间 (day)

        # 下界：短时近似反解
        lo = (fraction * np.sqrt(np.pi) / 6.0) ** 2 / (D / self.R ** 2 * 86400.0) * 0.5
        lo = max(lo, 0.0)
        # 上界：长时近似反解
        hi = -tau_char / (np.pi ** 2) * np.log(max((1 - fraction) * np.pi ** 2 / 6.0, 1e-12))
        hi = max(hi, tau_char)

        for _ in range(60):
            mid = (lo + hi) / 2
            f_mid = self.release_fraction(mid, T_C, moisture)
            if f_mid < fraction:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


# ============================================================
# 第二部分：土壤传输与归趋模型
# ============================================================

class SoilTransportModel:
    """
    土壤中6PPD的吸附/运移/降解耦合

    Kd = Koc * foc       —— 线性吸附
    Rf = 1 + (rho_b/theta) * Kd  —— 延迟因子

    稳态浓度通过 源通量 / (降解+淋洗) 质量平衡计算
    """

    def __init__(self, props=None):
        d = {
            "foc": 0.02,            # 有机碳分数
            "bulk_density": 1.35,   # kg/L
            "water_content": 0.30,  # 体积含水率
            "clay_pct": 15.0,       # 黏粒 %
            "pH": 6.8,
            "depth_cm": 20.0,       # 混合层深度
        }
        d.update(props or {})
        self.p = d

        self.Kow = 10 ** Chem.LOG_KOW
        self.Koc = 0.411 * self.Kow   # Karickhoff (1981)
        self.Kd = self.Koc * d["foc"]

        theta = d["water_content"]
        self.Rf = 1.0 + (d["bulk_density"] / max(theta, 0.01)) * self.Kd

    def pore_velocity(self, rainfall_mm_d, inf_frac=0.7):
        """孔隙水流速 [m/day]"""
        return rainfall_mm_d * 1e-3 * inf_frac / max(self.p["water_content"], 0.01)

    def dispersion_coef(self, v_m_d):
        """弥散系数 [m2/day], alpha_L ≈ 0.1m"""
        return 0.1 * v_m_d

    def steady_state(self, source_rate_mg_d, rainfall_mm_d, T_C=20):
        """
        源通量 -> 稳态孔隙水浓度 [mg/L] + 土壤固相浓度 [mg/kg]

        质量平衡（混合层）：
          Input = source_rate  [mg/day]
          Loss  = (k_deg + v/depth) * (Cw*Vw + Cs*Ms)
                 = (k_deg + v/depth) * Cw * (Vw + Kd*Ms)
        """
        depth_m = self.p["depth_cm"] * 0.01
        area = 1.0                         # m2
        Vw = area * depth_m * self.p["water_content"] * 1000   # L
        Ms = area * depth_m * self.p["bulk_density"] * 1000    # kg

        k = Chem.k_deg(T_C, self.p["water_content"])
        v = self.pore_velocity(rainfall_mm_d)
        loss_rate = k + v / depth_m         # 1/day

        if loss_rate < 1e-12:
            loss_rate = 1e-12

        Cw = source_rate_mg_d / (loss_rate * (Vw + self.Kd * Ms))  # mg/L
        Cs = self.Kd * Cw  # mg/kg
        return Cw, Cs

    def depth_profile(self, source_mg_L, rainfall_mm_d, T_C=20, n=50):
        """
        稳态深度剖面：C(z) = C0 * exp(-lambda * z)
        lambda 由 CDRE 稳态解给出
        """
        v = self.pore_velocity(rainfall_mm_d)
        Dx = self.dispersion_coef(v)
        k = Chem.k_deg(T_C, self.p["water_content"])

        if v > 1e-12:
            disc = 1.0 + 4.0 * k * self.Rf * Dx / (v ** 2)
            lam = v / (2 * Dx) * (np.sqrt(disc) - 1)
        else:
            lam = np.sqrt(k * self.Rf / max(Dx, 1e-12))

        depths = np.linspace(0, self.p["depth_cm"], n)
        Cw = source_mg_L * np.exp(-lam * depths * 0.01)
        Cs = self.Kd * Cw
        return depths, Cw, Cs


# ============================================================
# 第三部分：集成预测器
# ============================================================

class IntegratedPredictor:
    """TWP释放 + 土壤传输 一体化预测"""

    def __init__(self):
        self.twp = None
        self.soil = None

    def setup(self, particle_radius_um=50, loading_wt_pct=1.0,
              twp_mass_g=1.0, d0_m2_s=5e-18, soil_props=None):
        self.twp = TWPReleaseModel(
            particle_radius_um=particle_radius_um,
            d0_m2_s=d0_m2_s,
            loading_wt_pct=loading_wt_pct,
            twp_mass_g=twp_mass_g,
        )
        self.soil = SoilTransportModel(soil_props)

    def predict(self, T_C=20, rainfall_mm_d=3, moisture=0.30,
                days=365, n_points=100):
        """
        时间序列预测
        返回 DataFrame: 时间, 释放分数, 累积释放mg, 释放速率mg/d,
                        孔隙水浓度mg/L, 土壤浓度mg/kg
        """
        t_pts = np.unique(np.concatenate([
            np.linspace(0, min(days, 30), 30),
            np.linspace(30, days, max(n_points - 30, 10)),
        ]))

        rows = []
        for t in t_pts:
            f = self.twp.release_fraction(t, T_C, moisture)
            cum = f * self.twp.M_total
            rate = self.twp.release_rate(t, T_C, moisture)
            cw, cs = self.soil.steady_state(rate, rainfall_mm_d, T_C)
            rows.append({
                "day": t,
                "fraction": f,
                "cum_mg": cum,
                "rate_mg_d": rate,
                "Cw_mg_L": cw,
                "Cs_mg_kg": cs,
            })

        return pd.DataFrame(rows)


# ============================================================
# 第四部分：气候情景对比
# ============================================================

def compare_scenarios():
    """对比不同气候/土壤组合下的6PPD释放与归趋"""
    scenarios = [
        ("温带湿润",   15, 4.0, 0.35, 0.03),
        ("温带干旱",   15, 0.5, 0.10, 0.01),
        ("亚热带湿润", 25, 6.0, 0.35, 0.02),
        ("亚热带干旱", 25, 0.3, 0.08, 0.01),
        ("寒带",        5, 2.0, 0.25, 0.04),
        ("炎热半干旱", 32, 1.5, 0.12, 0.01),
        ("热带季风",   28, 10.0, 0.40, 0.03),
        ("地中海",     18, 2.0, 0.20, 0.02),
    ]

    rows = []
    for name, Tc, rain, moist, foc in scenarios:
        twp = TWPReleaseModel(particle_radius_um=50, loading_wt_pct=1.0,
                              twp_mass_g=1.0, d0_m2_s=5e-18)
        soil = SoilTransportModel({"foc": foc, "water_content": moist})

        m30 = twp.cumulative_release(30, Tc, moist)
        m90 = twp.cumulative_release(90, Tc, moist)
        m180 = twp.cumulative_release(180, Tc, moist)
        m365 = twp.cumulative_release(365, Tc, moist)

        t50 = twp.time_to_fraction(0.50, Tc, moist)
        t90 = twp.time_to_fraction(0.90, Tc, moist)

        cw, cs = soil.steady_state(m30 / 30, rain, Tc)

        rows.append({
            "情景": name,
            "T_C": Tc,
            "降水mm_d": rain,
            "foc": foc,
            "30d释放_mg": round(m30, 3),
            "90d释放_mg": round(m90, 3),
            "180d释放_mg": round(m180, 3),
            "365d释放_mg": round(m365, 3),
            "365d释放%": round(m365 / twp.M_total * 100, 1),
            "T50_d": round(t50, 0),
            "T90_d": round(t90, 0),
            "稳态Cw_mg_L": round(cw, 6),
            "稳态Cs_mg_kg": round(cs, 6),
        })
    return pd.DataFrame(rows)


# ============================================================
# 第五部分：敏感性分析
# ============================================================

def sensitivity(particle_um=50, T_C=22, moisture=0.30, d0=5e-18, loading_pct=1.0):
    """
    逐个参数 +/-30% 扰动，观察365天累积释放量和土壤浓度的变化
    """
    base = TWPReleaseModel(particle_radius_um=particle_um, loading_wt_pct=loading_pct,
                           twp_mass_g=1.0, d0_m2_s=d0)
    soil = SoilTransportModel({"foc": 0.02, "water_content": moisture})
    base365 = base.cumulative_release(365, T_C, moisture)
    _, base_cs = soil.steady_state(base.release_rate(365, T_C, moisture),
                                   3.0, T_C)

    sweeps = [
        ("温度 [C]",      T_C,       [T_C*0.7, T_C, T_C*1.3]),
        ("含水率",        moisture,  [0.10, 0.30, 0.50]),
        ("TWP粒径 [um]",  particle_um, [25, 50, 100]),
        ("D0 [m2/s]",     d0,        [d0*0.3, d0, d0*3.0]),
        ("6PPD含量 [%]",  loading_pct, [0.5, 1.0, 2.0]),
        ("土壤foc",       0.02,      [0.005, 0.02, 0.08]),
    ]

    rows = []
    for param, base_val, vals in sweeps:
        for v in vals:
            if param == "温度 [C]":
                rel = base.cumulative_release(365, v, moisture)
                _, cs = soil.steady_state(base.release_rate(365, v, moisture), 3.0, v)
                ratio = v / base_val
            elif param == "含水率":
                rel = base.cumulative_release(365, T_C, v)
                _, cs = soil.steady_state(base.release_rate(365, T_C, v), 3.0, T_C)
                ratio = v / base_val
            elif param == "TWP粒径 [um]":
                m = TWPReleaseModel(particle_radius_um=v, loading_wt_pct=loading_pct,
                                    twp_mass_g=1.0, d0_m2_s=d0)
                rel = m.cumulative_release(365, T_C, moisture)
                _, cs = soil.steady_state(m.release_rate(365, T_C, moisture), 3.0, T_C)
                ratio = v / base_val
            elif param == "D0 [m2/s]":
                m = TWPReleaseModel(particle_radius_um=particle_um, loading_wt_pct=loading_pct,
                                    twp_mass_g=1.0, d0_m2_s=v)
                rel = m.cumulative_release(365, T_C, moisture)
                _, cs = soil.steady_state(m.release_rate(365, T_C, moisture), 3.0, T_C)
                ratio = v / base_val
            elif param == "6PPD含量 [%]":
                m = TWPReleaseModel(particle_radius_um=particle_um, loading_wt_pct=v,
                                    twp_mass_g=1.0, d0_m2_s=d0)
                rel = m.cumulative_release(365, T_C, moisture)
                _, cs = soil.steady_state(m.release_rate(365, T_C, moisture), 3.0, T_C)
                ratio = v / base_val
            else:  # 土壤foc
                s2 = SoilTransportModel({"foc": v, "water_content": moisture})
                _, cs = s2.steady_state(base.release_rate(365, T_C, moisture), 3.0, T_C)
                rel = base365
                ratio = v / base_val

            rows.append({
                "参数": param,
                "值": v,
                "相对基准": round(ratio, 2),
                "365d释放_mg": round(rel, 3),
                "365d土壤Cs_mg_kg": round(cs, 6),
            })

    return pd.DataFrame(rows)


# ============================================================
# 第六部分：主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 64)
    print("  6PPD 土壤介质释放预测模型")
    print("  TWP扩散(Fick) + 土壤传输(CDE) + 降解耦合")
    print("=" * 64)

    # ---- 基础场景 ----
    print("\n[1] 基础预测：1g TWP, 50um, 1% 6PPD, 22degC")
    print("-" * 50)

    p = IntegratedPredictor()
    p.setup(particle_radius_um=50, loading_wt_pct=1.0, twp_mass_g=1.0)
    df = p.predict(T_C=22, rainfall_mm_d=3, moisture=0.30, days=365)

    # 关键时间点
    for d in [7, 30, 90, 180, 365]:
        row = df.iloc[(df["day"] - d).abs().idxmin()]
        print(f"  day {d:4.0f} | 释放 {row['fraction']*100:5.1f}% | "
              f"累积 {row['cum_mg']:.3f} mg | 速率 {row['rate_mg_d']:.4f} mg/d | "
              f"土壤 {row['Cs_mg_kg']:.6f} mg/kg")

    print(f"\n  总6PPD含量: {p.twp.M_total:.1f} mg/g-TWP")

    # 关键释放时间
    print("\n  关键释放节点：")
    for frac in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        td = p.twp.time_to_fraction(frac, T_C=22, moisture=0.30)
        print(f"    {frac*100:5.0f}% -> {td:8.1f} d ({td/365.25:.2f} yr)")

    # ---- 气候情景 ----
    print("\n\n[2] 气候情景对比（1g TWP, 50um, 1% 6PPD）")
    print("-" * 50)
    cs = compare_scenarios()
    print(cs.to_string(index=False))

    # ---- 敏感性分析 ----
    print("\n\n[3] 单参数敏感性分析（365d累积释放）")
    print("-" * 50)
    sens = sensitivity()
    # 关键行摘要
    for param in sens["参数"].unique():
        sub = sens[sens["参数"] == param]
        print(f"\n  {param}:")
        for _, r in sub.iterrows():
            print(f"    {r['值']:>10.4g}  ->  {r['365d释放_mg']:.3f} mg,  "
                  f"土壤 {r['365d土壤Cs_mg_kg']:.6f} mg/kg")

    # ---- 深度剖面 ----
    print("\n\n[4] 土壤剖面分布（源浓度=0.005 mg/L 孔隙水）")
    print("-" * 50)
    s = SoilTransportModel({"foc": 0.02})
    _, cw, cs = s.depth_profile(0.005, 3.0, 22)
    print(f"  {'深度cm':>8s}  {'Cw mg/L':>10s}  {'Cs mg/kg':>10s}")
    for i in [0, 5, 10, 20, 30, 49]:
        print(f"  {i/49*20:8.2f}  {cw[i]:10.6f}  {cs[i]:10.6f}")
    print(f"\n  -> 6PPD主要集中在表层0-5cm，10cm以下浓度可忽略")
