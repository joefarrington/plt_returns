import pandas as pd
import numpy as np
import jax.numpy as jnp


def real_input_df_to_array(
    df, max_demand, start_date="2021-01-01", period_split_hour=12
):
    # Converting real oberved demand and predictions from trained model
    # into input compatible with our environment
    df["day_counter"] = (
        pd.to_datetime(df["prediction_point_timestamp"]).dt.date
        - pd.to_datetime(start_date).date()
    ).dt.days
    df["period_counter"] = (
        pd.to_datetime(df["prediction_point_timestamp"]).dt.hour >= period_split_hour
    ).astype(int)
    days = []

    def get_elements(df_row, pred_returned_reversed=False):
        # If we predict not all units will be transfused, will issue YUFO
        # so want to have returns first (so will tx oldest units first)
        # If we predict all units will be tx, will issue OUFO so want to have
        # transfusions first (so again will tx oldest units first)
        total_requested = df_row["requested_quantity"]
        total_transfused = df_row["transfused_quantity"]
        prediction = df_row["prediction"]

        limit = min(total_requested, total_transfused)

        mask = [1] * total_requested
        returned = [0] * limit + [1] * (total_requested - limit)
        pred_returned = [prediction] * total_requested

        if pred_returned_reversed:
            mask.reverse()
            returned.reverse()
            pred_returned.reverse()

        return mask, returned, pred_returned

    for day in range(df["day_counter"].max() + 1):
        periods = []
        for period, temp_df in df[df["day_counter"] == day].groupby("period_counter"):
            mask = []
            returned = []
            pred_returned = []

            for _, row in temp_df.iterrows():
                m, r, pr = get_elements(row, row["prediction"] == 1)
                mask.extend(m)
                returned.extend(r)
                pred_returned.extend(pr)

            # Pad to consistent length
            fill_value = max_demand - len(mask)
            mask.extend([0] * fill_value)
            returned.extend([-1] * fill_value)
            pred_returned.extend([-1] * fill_value)
            periods.append([mask, returned, pred_returned])

        days.append(periods)
    return jnp.array(days)
