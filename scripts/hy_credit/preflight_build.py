"""Preflight rehearsal of the local post-pull stages on the REAL 2010 sample (no WRDS).

    uv run --with matplotlib python scripts/hy_credit/preflight_build.py   (from the repo root)

Lays the sample out as one development month under data/cache/_preflight (licensed rows
stay in data/cache), writes the Parquet files with the same Arrow schema the orchestrator
uses, then runs build-dev, report-dev, the plots and the final QA into a temporary
results directory, and deletes everything it created. Exit code 0 = all stages work.
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:

    from macro.data.hy_runs import _read_text, build_development, dev_report
    from macro.data.wrds import load_compustat_ratings, load_trace_master

    orchestrator = load("run_hy_once")
    cache = Path("data/cache/_preflight")
    results = Path(tempfile.mkdtemp(prefix="hy_preflight_results_"))
    shutil.rmtree(cache, ignore_errors=True)
    try:
        work = cache / "hy_credit"
        (work / "trace_raw").mkdir(parents=True)
        (cache / "fred").symlink_to(Path("data/cache/fred").resolve())
        raw = _read_text(Path("data/cache/trace_enhanced_sample.csv"))
        safe = orchestrator.server_side_keep(raw)
        raw = raw.loc[safe, orchestrator.RAW_COLUMNS]
        # Same Arrow schema and conversion path the orchestrator uses for real chunks.
        orchestrator.write_parquet_frame(
            raw, work / "trace_raw" / "2010-01.parquet", orchestrator.RAW_SCHEMA
        )
        load_trace_master(Path("data/cache/trace_master_sample.csv")).to_parquet(
            work / "master.parquet"
        )
        _read_text(Path("data/cache/comp_security_sample.csv")).to_parquet(
            work / "security.parquet"
        )
        _read_text(Path("data/cache/bondcrsp_link_sample.csv")).to_parquet(
            work / "bondcrsp.parquet"
        )
        load_compustat_ratings(Path("data/cache/compustat_ratings_sample.csv")).to_parquet(
            work / "ratings.parquet"
        )
        daily = build_development(cache=cache, out_dir=results)
        assert daily["n_hy_issuers"].max() >= 10, "sample build produced no HY aggregate"
        dev_report(cache=cache, out_dir=results)
        plots = load("plot_hy").make_plots(work / "hy_daily.parquet", results)
        assert all(p.exists() and p.stat().st_size > 10_000 for p in plots), "plot missing/empty"
        code = load("qa_dev").main(cache, results)
        assert code == 0, "QA reported failures on the sample build"
        print(f"PREFLIGHT BUILD OK: {len(daily)} daily rows, plots {[p.name for p in plots]}")
        return 0
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(results, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
