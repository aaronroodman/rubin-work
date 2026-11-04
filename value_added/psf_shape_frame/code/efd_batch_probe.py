"""Probe whether the EFD answers from the current node; prints the row count."""
import asyncio
import socket

from astropy.time import Time

from rubinwork.common.telemetry_clients import make_efd_client


async def main():
    client = make_efd_client()
    df = await client.select_time_series(
        'lsst.sal.ESS.airTurbulence', ['salIndex', 'speed0'],
        Time('2026-11-03T03:00', scale='utc'), Time('2026-11-03T03:02', scale='utc'))
    print(socket.gethostname(), 'EFD rows', len(df), flush=True)


asyncio.run(main())
