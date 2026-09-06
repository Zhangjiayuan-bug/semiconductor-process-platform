from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 360
    equipment = rng.choice(["ETCH-01", "ETCH-02", "ETCH-03"], n, p=[0.4, 0.35, 0.25])
    step = rng.choice(["etch", "deposition", "clean"], n, p=[0.5, 0.3, 0.2])
    temperature = rng.normal(350, 7, n) + np.where(equipment == "ETCH-03", 5, 0)
    pressure = rng.normal(1.20, 0.08, n)
    flow = rng.normal(80, 3, n)
    thickness = 100 + 0.25 * (temperature - 350) - 3 * (pressure - 1.2) + rng.normal(0, 1.5, n)
    yield_pct = 96 - 0.10 * abs(temperature - 350) - 2.4 * abs(pressure - 1.2) + rng.normal(0, 1.2, n)
    yield_pct += np.where(equipment == "ETCH-03", -2.0, 0)
    yield_pct[::97] -= 10
    data = pd.DataFrame(
        {
            "lot_id": [f"LOT-{i // 12 + 1:03d}" for i in range(n)],
            "wafer_id": [f"W{i + 1:04d}" for i in range(n)],
            "equipment": equipment,
            "process_step": step,
            "temperature_c": temperature.round(2),
            "pressure_kpa": pressure.round(3),
            "gas_flow_sccm": flow.round(2),
            "thickness_nm": thickness.round(2),
            "yield_pct": yield_pct.clip(70, 100).round(2),
            "defect_count": rng.poisson(3, n),
        }
    )
    output = Path(__file__).parent / "data" / "demo_process_data.csv"
    output.parent.mkdir(exist_ok=True)
    data.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
