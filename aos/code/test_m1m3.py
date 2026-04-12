#!/usr/bin/env python3
"""Quick test of M1M3 thermal gradient retrieval from command line."""

import asyncio
import numpy as np
import pandas as pd
from astropy.time import Time

# Test imports
print("1. Testing imports...")
try:
    from lsst.ts.m1m3.utils import ThermocoupleAnalysis
    print("   ThermocoupleAnalysis: OK")
except ImportError as e:
    print(f"   ThermocoupleAnalysis: FAILED - {e}")
    raise SystemExit(1)

try:
    from lsst.summit.utils.efdUtils import makeEfdClient
    print("   makeEfdClient: OK")
except ImportError as e:
    print(f"   makeEfdClient: FAILED - {e}")
    raise SystemExit(1)


async def test_gradients():
    # Use a known good time range (one FAM visit on 20260315)
    start = Time("2025-03-16T03:00:00", scale="utc")
    end = Time("2025-03-16T04:00:00", scale="utc")

    print(f"\n2. Creating EFD client...")
    efd_client = makeEfdClient()
    print(f"   EFD client: {type(efd_client)}")

    print(f"\n3. Testing raw EFD query (thermocouple topic)...")
    try:
        data = await efd_client.select_time_series(
            "lsst.sal.MTM1M3TS.thermocoupleScannerInfo",
            [], start, end,
        )
        if data is not None and len(data) > 0:
            print(f"   Raw query: OK, {len(data)} rows")
        else:
            print("   Raw query: returned no data")
    except Exception as e:
        print(f"   Raw query: FAILED - {type(e).__name__}: {e}")

    print(f"\n4. Testing ThermocoupleAnalysis.load()...")
    try:
        tc = ThermocoupleAnalysis(efd_client)
        await tc.load(start, end, time_bin=30)
        gradients = tc.xyz_r_gradients
        print(f"   load: OK, {len(gradients)} gradient rows")
        print(f"   columns: {list(gradients.columns)}")
        print(f"   index type: {type(gradients.index)}")
        if len(gradients) > 0:
            print(f"   first row: {gradients.iloc[0].to_dict()}")
    except Exception as e:
        import traceback
        print(f"   load: FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()

    print(f"\n5. Testing full interpolation pipeline...")
    try:
        # Simulate a visit_table with one obs_start
        visit_table = pd.DataFrame({
            "obs_start": ["2025-03-16T03:30:00"],
            "day_obs": [20250316],
            "seq_num": [100],
        })
        date_strings = Time(
            [str(x) for x in visit_table["obs_start"].values],
            format="isot", scale="tai"
        ).utc.isot
        data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True)
        sorted_data_times = data_times.sort_values()
        print(f"   sorted_data_times type: {type(sorted_data_times)}")
        print(f"   [0] access: {sorted_data_times[0]}")
        print(f"   [-1] access: {sorted_data_times[-1]}")

        grad_times = pd.to_datetime(
            gradients.index, format="ISO8601", utc=True
        ).astype("int64")
        data_times_int = data_times.astype("int64")
        t0 = grad_times[0]
        grad_times_norm = (grad_times - t0) / 1e9
        data_times_norm = (data_times_int - t0) / 1e9

        for name in ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]:
            values = gradients[name].values
            val_interp = pd.Series(values).interpolate().values
            result = np.interp(data_times_norm, grad_times_norm, val_interp)
            print(f"   {name}: {result[0]:.4f}")

        print("   Interpolation: OK")
    except Exception as e:
        import traceback
        print(f"   Interpolation: FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_gradients())
