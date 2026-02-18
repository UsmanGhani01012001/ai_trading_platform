"""
AWM AI Background Writer
========================
Background thread that periodically writes bars to Parquet files.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from time import sleep
from pathlib import Path
import logging

from app.queue import drain_bars

logger = logging.getLogger(__name__)


def get_parquet_path(instrument: str = "NQ", series: str = "50-50-75") -> str:
    """Get parquet file path for instrument/series"""
    base_dir = Path("data/awm") / instrument
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / f"AWM_{series}.parquet")


def parquet_writer_loop(
    default_instrument: str = "NQ",
    default_series: str = "50-50-75",
    write_interval: float = 0.5
):
    """
    Background loop that writes bars to Parquet.
    
    Args:
        default_instrument: Default instrument if not in bar data
        default_series: Default series if not in bar data
        write_interval: Seconds between writes (default 500ms)
    """
    logger.info(f"Background writer started (interval={write_interval}s)")
    
    while True:
        try:
            bars = drain_bars()
            
            if bars:
                # Group by instrument/series
                groups = {}
                for bar in bars:
                    key = (
                        bar.get('instrument', default_instrument),
                        bar.get('series', default_series)
                    )
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(bar)
                
                # Write each group to its parquet file
                for (instrument, series), group_bars in groups.items():
                    parquet_path = get_parquet_path(instrument, series)
                    
                    df = pd.DataFrame(group_bars)
                    table = pa.Table.from_pandas(df)
                    
                    # Append if file exists, create otherwise
                    if os.path.exists(parquet_path):
                        # Read existing, append, write
                        existing = pq.read_table(parquet_path)
                        combined = pa.concat_tables([existing, table])
                        pq.write_table(combined, parquet_path, compression="snappy")
                    else:
                        pq.write_table(table, parquet_path, compression="snappy")
                    
                    logger.debug(f"Wrote {len(group_bars)} bars to {parquet_path}")
            
            sleep(write_interval)
            
        except Exception as e:
            logger.error(f"Background writer error: {e}")
            sleep(1)  # Back off on error