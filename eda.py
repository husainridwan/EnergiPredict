import matplotlib.pyplot as plt
import pandas as pd

from energipredict.data import load_dataset


def load_balance(df: pd.DataFrame) -> None:
    print("=== HVAC load summary ===")
    print(df["hvac_total"].describe().to_string())
    print(f"\nMean load: {df['hvac_total'].mean():.1f} kW, peak: {df['hvac_total'].max():.1f} kW")
    print(f"Total energy over this window: {df['hvac_total'].sum():,.0f} kWh")
    print()


def weather_drivers(df: pd.DataFrame) -> None:
    print("=== What drives load: correlation with weather ===")
    corr = df[["hvac_total", "air_temp_set_1", "relative_humidity_set_1",
               "solar_radiation_set_1"]].corr()["hvac_total"]
    print(corr.drop("hvac_total").sort_values(key=abs, ascending=False).to_string())
    print()


def occupancy_pattern(df: pd.DataFrame, show_plot: bool = True) -> None:
    print("=== Load by hour of day, weekday vs weekend ===")
    df2 = df.copy()
    df2["hour"] = df2.index.hour
    df2["is_weekend"] = df2.index.dayofweek >= 5
    by_hour = df2.groupby(["hour", "is_weekend"])["hvac_total"].mean().unstack()
    by_hour.columns = ["Weekday", "Weekend"]
    print(by_hour.round(1).to_string())
    print()

    if show_plot:
        fig, ax = plt.subplots(figsize=(9, 4))
        by_hour.plot(ax=ax)
        ax.set_title("Mean HVAC load by hour of day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("kW")
        plt.tight_layout()
        plt.savefig("reports/figures/load_by_hour.png", dpi=120)
        print("Saved reports/figures/load_by_hour.png")


def main():
    df, report = load_dataset()
    print(report.summary())
    print()
    load_balance(df)
    weather_drivers(df)

    from pathlib import Path
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    occupancy_pattern(df)


if __name__ == "__main__":
    main()
