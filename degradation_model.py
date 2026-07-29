"""6PPD/6PPD-Q 在土壤和水体中的条件化动态归趋模型。

该版本优先补足机理预测所需的主干缺口：
- 土壤氧化还原条件从二分类扩展到好氧、普通厌氧、硝酸盐还原、硫酸盐还原、铁还原、灭菌等情景；
- 水体过程拆分为 6PPD 消散、6PPD-Q 生成和 6PPD-Q 光化学/间接光化学消散；
- 模拟使用一级反应解析步进，避免显式 Euler 步长过大时出现数值偏差；
- 输出机理特征，便于后续训练时作为机器学习模型的可解释输入。

默认动力学参数主要采用 Shen et al. (2025) 的土壤培养结果，并用 Wang et al. (2026)
和中文综述中的水体光化学机制作为先验结构。所有参数都应在本地实验数据充足后校准。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Medium = Literal["soil", "water"]
Redox = Literal[
    "aerobic",
    "anaerobic",
    "nitrate_reducing",
    "sulfate_reducing",
    "iron_reducing",
    "sterilized",
]


@dataclass(frozen=True)
class FateConditions:
    """模型所需环境条件，单位见字段名。"""

    medium: Medium = "soil"
    redox: Redox = "aerobic"
    temperature_c: float = 22.0
    pH: float = 7.0
    moisture: float = 0.30
    eh_mv: float | None = None
    organic_carbon_fraction: float = 0.02
    bulk_density_kg_l: float = 1.35
    water_volume_l: float = 60.0
    soil_mass_kg: float = 270.0
    light_factor: float = 0.0
    ozone_factor: float = 0.0
    epfr_factor: float = 0.0
    nom_mg_l: float = 0.0
    nitrate_mmol_l: float = 0.0


@dataclass(frozen=True)
class KineticParameters:
    """一级速率参数；所有速率单位为 day^-1，半衰期单位为 day。"""

    activation_energy_j_mol: float = 50_000.0
    reference_temperature_c: float = 22.0

    soil_6ppd_half_life_aerobic_d: float = 1.1
    soil_6ppd_half_life_sterilized_aerobic_d: float = 1.8
    soil_6ppd_half_life_anaerobic_d: float = 51.4
    soil_6ppd_half_life_reducing_floor_d: float = 120.0

    soil_6ppdq_half_life_aerobic_d: float = 13.85
    soil_6ppdq_half_life_anaerobic_d: float = 21.1
    soil_6ppdq_half_life_nitrate_reducing_d: float = 28.1
    soil_6ppdq_half_life_sulfate_reducing_d: float = 26.6
    soil_6ppdq_half_life_iron_reducing_d: float = 21.3

    water_6ppd_half_life_dark_d: float = 0.30
    water_6ppd_half_life_light_d: float = 5.0 / 24.0
    water_6ppdq_half_life_dark_d: float = 17.5 / 24.0
    water_6ppdq_half_life_light_d: float = 11.2 / 24.0
    water_6ppdq_half_life_nitrate_light_d: float = 2.6 / 24.0

    q_formation_yield_water: float = 0.03
    q_formation_yield_soil: float = 0.002
    q_formation_yield_flooded_iron: float = 0.02

    koc_6ppd_l_kg: float = 10 ** 4.84
    koc_6ppdq_l_kg: float = 10 ** 3.93


def half_life_to_rate(half_life_d: float | None) -> float:
    """将半衰期转换为一级速率常数；None 表示无显著转化。"""
    if half_life_d is None:
        return 0.0
    if half_life_d <= 0:
        raise ValueError("半衰期必须大于 0")
    return float(np.log(2.0) / half_life_d)


def temperature_adjusted_rate(rate_ref: float, temperature_c: float, reference_c: float, ea: float) -> float:
    """按 Arrhenius 方程修正一级速率。"""
    if rate_ref <= 0:
        return 0.0
    gas_constant = 8.314
    temperature_k = temperature_c + 273.15
    reference_k = reference_c + 273.15
    if temperature_k <= 0:
        raise ValueError("温度必须高于绝对零度")
    return float(rate_ref * np.exp(ea / gas_constant * (1.0 / reference_k - 1.0 / temperature_k)))


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class TwoSpeciesFateModel:
    """追踪 6PPD 与 6PPD-Q 总质量、分配和条件化降解的动态模型。"""

    def __init__(self, conditions: FateConditions, parameters: KineticParameters | None = None):
        self.conditions = conditions
        self.parameters = parameters or KineticParameters()

    def _soil_half_lives(self) -> tuple[float | None, float | None]:
        p = self.parameters
        redox = self.conditions.redox
        if redox == "aerobic":
            return p.soil_6ppd_half_life_aerobic_d, p.soil_6ppdq_half_life_aerobic_d
        if redox == "sterilized":
            return p.soil_6ppd_half_life_sterilized_aerobic_d, None
        if redox == "nitrate_reducing":
            return p.soil_6ppd_half_life_reducing_floor_d, p.soil_6ppdq_half_life_nitrate_reducing_d
        if redox == "sulfate_reducing":
            return p.soil_6ppd_half_life_reducing_floor_d, p.soil_6ppdq_half_life_sulfate_reducing_d
        if redox == "iron_reducing":
            return p.soil_6ppd_half_life_reducing_floor_d, p.soil_6ppdq_half_life_iron_reducing_d
        return p.soil_6ppd_half_life_anaerobic_d, p.soil_6ppdq_half_life_anaerobic_d

    def _water_half_lives(self) -> tuple[float, float]:
        c = self.conditions
        p = self.parameters
        light = _clip01(c.light_factor)
        six_half_life = (
            (1.0 - light) * p.water_6ppd_half_life_dark_d
            + light * p.water_6ppd_half_life_light_d
        )
        if light > 0 and c.nitrate_mmol_l >= 1.0:
            q_half_life = p.water_6ppdq_half_life_nitrate_light_d
        else:
            q_half_life = (
                (1.0 - light) * p.water_6ppdq_half_life_dark_d
                + light * p.water_6ppdq_half_life_light_d
            )
        return six_half_life, q_half_life

    def _formation_rate(self, six_loss_rate: float) -> float:
        c = self.conditions
        p = self.parameters
        light = _clip01(c.light_factor)
        ozone = _clip01(c.ozone_factor)
        epfr = _clip01(c.epfr_factor)
        if c.medium == "water":
            enhancement = 0.25 + 0.75 * light + 0.20 * min(c.nom_mg_l, 20.0) / 20.0
            return six_loss_rate * p.q_formation_yield_water * enhancement
        if c.redox == "iron_reducing":
            yield_fraction = p.q_formation_yield_flooded_iron
        elif c.redox in {"anaerobic", "nitrate_reducing", "sulfate_reducing"}:
            yield_fraction = p.q_formation_yield_soil * (1.0 + 2.0 * epfr)
        else:
            yield_fraction = p.q_formation_yield_soil * (1.0 + ozone + epfr)
        return six_loss_rate * yield_fraction

    def _base_rates(self) -> tuple[float, float, float]:
        c = self.conditions
        p = self.parameters
        if c.medium == "soil":
            six_half_life, q_half_life = self._soil_half_lives()
        else:
            six_half_life, q_half_life = self._water_half_lives()

        six_rate = temperature_adjusted_rate(
            half_life_to_rate(six_half_life),
            c.temperature_c,
            p.reference_temperature_c,
            p.activation_energy_j_mol,
        )
        q_rate = temperature_adjusted_rate(
            half_life_to_rate(q_half_life),
            c.temperature_c,
            p.reference_temperature_c,
            p.activation_energy_j_mol,
        )
        if c.medium == "water":
            light = _clip01(c.light_factor)
            q_rate *= 1.0 + 0.05 * c.nom_mg_l * light
            q_rate *= 1.0 + 0.15 * c.nitrate_mmol_l * light
        return six_rate, q_rate, self._formation_rate(six_rate)

    def rates(self) -> dict[str, float]:
        """返回条件下的速率常数，方便校准和记录。"""
        six_rate, q_rate, formation_rate = self._base_rates()
        return {
            "k_6ppd_loss_d-1": six_rate,
            "k_6ppdq_loss_d-1": q_rate,
            "k_6ppd_to_q_d-1": formation_rate,
            "k_6ppd_total_d-1": six_rate + formation_rate,
        }

    def mechanistic_features(self) -> dict[str, float]:
        """生成后续机器学习训练可直接使用的机理特征。"""
        c = self.conditions
        p = self.parameters
        six_rate, q_rate, formation_rate = self._base_rates()
        kd_6ppd = p.koc_6ppd_l_kg * c.organic_carbon_fraction if c.medium == "soil" else 0.0
        kd_6ppdq = p.koc_6ppdq_l_kg * c.organic_carbon_fraction if c.medium == "soil" else 0.0
        capacity_6ppd, capacity_6ppdq, _, _ = self._distribution_capacity()
        return {
            "mech_k_6ppd_loss_d-1": six_rate,
            "mech_k_6ppdq_loss_d-1": q_rate,
            "mech_k_6ppd_to_q_d-1": formation_rate,
            "mech_kd_6ppd_l_kg": kd_6ppd,
            "mech_kd_6ppdq_l_kg": kd_6ppdq,
            "mech_capacity_6ppd_l": capacity_6ppd,
            "mech_capacity_6ppdq_l": capacity_6ppdq,
            "mech_light_factor": _clip01(c.light_factor),
            "mech_ozone_factor": _clip01(c.ozone_factor),
            "mech_epfr_factor": _clip01(c.epfr_factor),
        }

    def _distribution_capacity(self) -> tuple[float, float, float, float]:
        c = self.conditions
        p = self.parameters
        if c.medium == "water":
            return c.water_volume_l, c.water_volume_l, 0.0, 0.0
        kd_6ppd = p.koc_6ppd_l_kg * c.organic_carbon_fraction
        kd_6ppdq = p.koc_6ppdq_l_kg * c.organic_carbon_fraction
        capacity_6ppd = c.water_volume_l + kd_6ppd * c.soil_mass_kg
        capacity_6ppdq = c.water_volume_l + kd_6ppdq * c.soil_mass_kg
        return capacity_6ppd, capacity_6ppdq, kd_6ppd, kd_6ppdq

    @staticmethod
    def _decay_integral(rate: float, step_d: float) -> float:
        """Integral of exp(-rate * t) from 0 to step_d."""
        if rate <= 0:
            return step_d
        return float((1.0 - np.exp(-rate * step_d)) / rate)

    @staticmethod
    def _bateman_integral(source_rate: float, sink_rate: float, step_d: float) -> float:
        """Integral of exp(-source_rate * t) * exp(-sink_rate * (step_d - t)) dt."""
        if abs(sink_rate - source_rate) < 1e-12:
            return float(step_d * np.exp(-sink_rate * step_d))
        return float(
            (np.exp(-source_rate * step_d) - np.exp(-sink_rate * step_d))
            / (sink_rate - source_rate)
        )

    @staticmethod
    def _step_reactions(
        mass_6ppd: float,
        mass_6ppdq: float,
        input_6ppd_mg_d: float,
        six_loss_rate: float,
        q_loss_rate: float,
        formation_rate: float,
        step_d: float,
    ) -> tuple[float, float]:
        total_six_rate = six_loss_rate + formation_rate
        if total_six_rate > 0:
            next_6ppd = mass_6ppd * np.exp(-total_six_rate * step_d)
            next_6ppd += input_6ppd_mg_d * (1.0 - np.exp(-total_six_rate * step_d)) / total_six_rate
        else:
            next_6ppd = mass_6ppd + input_6ppd_mg_d * step_d

        initial_6ppd_contribution = mass_6ppd * TwoSpeciesFateModel._bateman_integral(
            total_six_rate, q_loss_rate, step_d
        )
        if total_six_rate > 0:
            input_6ppd_contribution = input_6ppd_mg_d / total_six_rate * (
                TwoSpeciesFateModel._decay_integral(q_loss_rate, step_d)
                - TwoSpeciesFateModel._bateman_integral(total_six_rate, q_loss_rate, step_d)
            )
        else:
            if q_loss_rate > 0:
                input_6ppd_contribution = input_6ppd_mg_d * (
                    step_d / q_loss_rate
                    - (1.0 - np.exp(-q_loss_rate * step_d)) / (q_loss_rate ** 2)
                )
            else:
                input_6ppd_contribution = 0.5 * input_6ppd_mg_d * step_d ** 2
        formed_q = formation_rate * (initial_6ppd_contribution + input_6ppd_contribution)

        if q_loss_rate > 0:
            next_6ppdq = mass_6ppdq * np.exp(-q_loss_rate * step_d) + formed_q
        else:
            next_6ppdq = mass_6ppdq + formation_rate * mass_6ppd * step_d
        return max(float(next_6ppd), 0.0), max(float(next_6ppdq), 0.0)

    def simulate(
        self,
        days: float,
        initial_6ppd_mg: float,
        initial_6ppdq_mg: float = 0.0,
        input_6ppd_mg_d: float = 0.0,
        step_d: float = 0.1,
    ) -> pd.DataFrame:
        """模拟浓度时间序列；输入项可代表 TWP 持续释放、径流补给或外源负荷。"""
        if days <= 0 or step_d <= 0:
            raise ValueError("days 和 step_d 必须大于 0")
        if min(initial_6ppd_mg, initial_6ppdq_mg, input_6ppd_mg_d) < 0:
            raise ValueError("初始质量和输入速率不能为负")

        six_rate, q_rate, formation_rate = self._base_rates()
        capacity_6ppd, capacity_6ppdq, kd_6ppd, kd_6ppdq = self._distribution_capacity()
        times = np.arange(0.0, days + step_d, step_d)
        mass_6ppd = float(initial_6ppd_mg)
        mass_6ppdq = float(initial_6ppdq_mg)
        rows = []

        for time_d in times:
            dissolved_6ppd = mass_6ppd / capacity_6ppd
            dissolved_6ppdq = mass_6ppdq / capacity_6ppdq
            rows.append(
                {
                    "time_d": float(time_d),
                    "6PPD_total_mg": mass_6ppd,
                    "6PPD_Q_total_mg": mass_6ppdq,
                    "6PPD_dissolved_mg_L": dissolved_6ppd,
                    "6PPD_Q_dissolved_mg_L": dissolved_6ppdq,
                    "6PPD_sorbed_mg_kg": kd_6ppd * dissolved_6ppd if self.conditions.medium == "soil" else 0.0,
                    "6PPD_Q_sorbed_mg_kg": kd_6ppdq * dissolved_6ppdq if self.conditions.medium == "soil" else 0.0,
                    **self.rates(),
                }
            )
            if time_d >= days:
                continue
            effective_step = min(step_d, days - time_d)
            mass_6ppd, mass_6ppdq = self._step_reactions(
                mass_6ppd,
                mass_6ppdq,
                input_6ppd_mg_d,
                six_rate,
                q_rate,
                formation_rate,
                effective_step,
            )
        return pd.DataFrame(rows)
