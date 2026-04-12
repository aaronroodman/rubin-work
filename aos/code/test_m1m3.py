#!/usr/bin/env python3
"""Quick test of M1M3 thermal gradient retrieval from command line."""

import asyncio
import os
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

try:
    from lsst_efd_client import EfdClient
    print("   EfdClient: OK")
except ImportError as e:
    print(f"   EfdClient: FAILED - {e}")


async def test_gradients():
    start = Time("2026-03-16T03:00:00", scale="utc")
    end = Time("2026-03-16T04:00:00", scale="utc")

    print(f"\n2. Creating EFD client...")
    efd_client = makeEfdClient()
    print(f"   EFD client type: {type(efd_client)}")

    # Check what instance/URL the client is using
    for attr in ['_url', 'url', '_client', 'influx_client', '_db']:
        if hasattr(efd_client, attr):
            print(f"   {attr}: {getattr(efd_client, attr)}")

    # Check environment variables that might affect EFD connection
    for var in ['EFD_ENDPOINT', 'LSST_EFD_ENDPOINT', 'EFD_HOST',
                'LSST_DDS_PARTITION_PREFIX']:
        val = os.environ.get(var)
        if val:
            print(f"   env {var}: {val}")

    print(f"\n3. Listing available topics (first 5 matching MTM1M3)...")
    try:
        topics = await efd_client.get_topics()
        m1m3_topics = [t for t in topics if 'MTM1M3' in t]
        print(f"   Total topics: {len(topics)}, MTM1M3 topics: {len(m1m3_topics)}")
        for t in m1m3_topics[:10]:
            print(f"     {t}")
    except Exception as e:
        print(f"   get_topics: FAILED - {type(e).__name__}: {e}")

    print(f"\n4. Testing simple EFD query (ESS temperature, known working)...")
    try:
        data = await efd_client.select_time_series(
            "lsst.sal.ESS.temperature",
            ["temperatureItem0"], start, end,
            index=113,
        )
        if data is not None and len(data) > 0:
            print(f"   ESS query: OK, {len(data)} rows")
        else:
            print("   ESS query: returned no data")
    except Exception as e:
        print(f"   ESS query: FAILED - {type(e).__name__}: {e}")

    print(f"\n5. Testing thermocouple topic query...")
    try:
        data = await efd_client.select_time_series(
            "lsst.sal.MTM1M3TS.thermocoupleScannerInfo",
            ["thermocouple0"], start, end,
        )
        if data is not None and len(data) > 0:
            print(f"   Thermocouple query: OK, {len(data)} rows")
        else:
            print("   Thermocouple query: returned no data")
    except Exception as e:
        print(f"   Thermocouple query: FAILED - {type(e).__name__}: {e}")

    print(f"\n6. Testing ThermocoupleAnalysis.load()...")
    try:
        tc = ThermocoupleAnalysis(efd_client)
        await tc.load(start, end, time_bin=30)
        gradients = tc.xyz_r_gradients
        if gradients is not None:
            print(f"   load: OK, {len(gradients)} gradient rows")
        else:
            print("   load: returned None (no data or query failed silently)")
    except Exception as e:
        import traceback
        print(f"   load: FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_gradients())
