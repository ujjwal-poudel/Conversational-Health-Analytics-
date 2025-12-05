"""
Parallel OpenL3 extraction runner.

Runs the existing OpenL3 extractor across multiple participants in parallel
using multiprocessing. Does NOT modify any extraction logic - just parallelizes.

Usage:
    python -m src.features.parallel_openl3_runner --workers 2
"""

import os
import argparse
from multiprocessing import Pool

from src.utils import config
from src.utils.logging_utils import get_logger
from src.features.openl3_extractor import OpenL3Extractor
import numpy as np

logger = get_logger(__name__)

# Global extractor - initialized once per worker process
_extractor = None


def init_worker():
    """
    Initializer function called once when each worker process starts.
    Loads the OpenL3 model once per worker, not per file.
    """
    global _extractor
    _extractor = OpenL3Extractor()


def extract_single_participant(args):
    """
    Extract OpenL3 embeddings for a single participant.
    Uses the pre-loaded model from init_worker().
    """
    global _extractor
    pid, wav_path, out_path = args
    
    # Skip if already exists
    if os.path.exists(out_path):
        return f"SKIP: {pid} (already exists)"
    
    try:
        emb = _extractor.extract_embeddings(wav_path)
        
        if emb is None:
            return f"FAIL: {pid} (extraction returned None)"
        
        np.save(out_path, emb)
        return f"OK: {pid} -> {out_path}"
    
    except Exception as e:
        return f"ERROR: {pid} - {str(e)}"


def run_parallel_extraction(num_workers=2):
    """
    Runs OpenL3 extraction in parallel across participants.
    
    Parameters
    ----------
    num_workers : int
        Number of parallel workers. Default 2 to avoid memory issues.
        Recommended: 2-3 for 16GB RAM, 1-2 for 8GB RAM.
    """
    processed_dir = config.PROCESSED_AUDIO_DIR
    output_dir = config.FEATURE_OPENL3_DIR
    
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build list of work items
    wav_files = [f for f in os.listdir(processed_dir) if f.endswith(".wav")]
    
    work_items = []
    for fname in wav_files:
        pid = fname.replace(".wav", "")
        wav_path = os.path.join(processed_dir, fname)
        out_path = os.path.join(output_dir, f"{pid}.npy")
        work_items.append((pid, wav_path, out_path))
    
    logger.info(f"Found {len(work_items)} audio files to process")
    logger.info(f"Using {num_workers} parallel workers")
    
    # Run in parallel with initializer (model loads once per worker)
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.map(extract_single_participant, work_items)
    
    # Log results
    ok_count = sum(1 for r in results if r.startswith("OK"))
    skip_count = sum(1 for r in results if r.startswith("SKIP"))
    fail_count = sum(1 for r in results if r.startswith("FAIL") or r.startswith("ERROR"))
    
    logger.info(f"Parallel extraction complete: {ok_count} OK, {skip_count} skipped, {fail_count} failed")
    
    for result in results:
        if result.startswith("ERROR") or result.startswith("FAIL"):
            logger.warning(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel OpenL3 extraction")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    args = parser.parse_args()
    
    run_parallel_extraction(num_workers=args.workers)
