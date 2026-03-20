"""
src/model_store.py
S3 Model Store for ARIA

Handles uploading and downloading ML model files to/from AWS S3.

Architecture:
    Local training  → upload to S3  → Railway downloads on startup
    
S3 bucket structure:
    aria-models/
        xgboost/
            model.ubj           XGBoost model binary
            metadata.json       Training metrics and timestamp
        prophet/
            electronics.pkl     Prophet model per category
            fashion.pkl
            home_goods.pkl
            sports.pkl
            metadata.json       Training metrics per category

Why S3:
    Model files are binary artifacts — they don't belong in git.
    S3 gives us versioned, durable, globally accessible storage.
    The same boto3 code works with AWS S3, Cloudflare R2, and
    any S3-compatible storage — one interface, multiple providers.

Usage:
    # Upload after training locally
    python src/model_store.py --upload

    # Download on Railway startup
    python src/model_store.py --download

    # Check what's in S3
    python src/model_store.py --status
"""
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("model_store")


# ── Configuration ─────────────────────────────────────────────────────

def get_s3_config():
    """Load S3 configuration from settings."""
    from config.settings import get_settings
    settings = get_settings()
    return {
        "bucket":     settings.s3_bucket_name,
        "region":     settings.s3_region,
        "access_key": settings.aws_access_key_id,
        "secret_key": settings.aws_secret_access_key,
        "prefix":     settings.s3_model_prefix,
    }


def get_s3_client():
    """Create and return an S3 client."""
    try:
        import boto3
        config = get_s3_config()
        client = boto3.client(
            "s3",
            region_name=config["region"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
        )
        return client, config
    except ImportError:
        log.error("boto3 not installed. Run: pip install boto3")
        raise


# ── Model file definitions ────────────────────────────────────────────

def get_model_files():
    """
    Returns all model files with their local paths and S3 keys.
    Add new model files here as the project grows.
    """
    settings_module = ROOT / "config" / "settings.py"
    from config.settings import get_settings
    settings = get_settings()

    files = []

    # XGBoost pricing model
    xgb_model = settings.xgb_model_path
    xgb_meta  = settings.xgb_meta_path

    if xgb_model.exists():
        files.append({
            "local": xgb_model,
            "s3_key": "xgboost/model.ubj",
            "description": "XGBoost pricing model"
        })
    if xgb_meta.exists():
        files.append({
            "local": xgb_meta,
            "s3_key": "xgboost/metadata.json",
            "description": "XGBoost training metadata"
        })

    # Prophet demand forecasting models
    prophet_dir = settings.prophet_dir
    for category in ["electronics", "fashion", "home_goods", "sports"]:
        model_file = prophet_dir / f"prophet_{category}.pkl"
        meta_file  = prophet_dir / f"prophet_{category}_meta.json"
        if model_file.exists():
            files.append({
                "local": model_file,
                "s3_key": f"prophet/{category}.pkl",
                "description": f"Prophet model - {category}"
            })
        if meta_file.exists():
            files.append({
                "local": meta_file,
                "s3_key": f"prophet/{category}_meta.json",
                "description": f"Prophet metadata - {category}"
            })

    return files


# ── Upload ────────────────────────────────────────────────────────────

def upload_models(dry_run: bool = False) -> dict:
    """
    Upload all local model files to S3.
    Called after training locally — makes models available to Railway.

    Returns summary of what was uploaded.
    """
    client, config = get_s3_client()
    bucket  = config["bucket"]
    prefix  = config["prefix"]
    files   = get_model_files()

    if not files:
        log.warning("No model files found locally. Train models first:")
        log.warning("  python src/features.py")
        log.warning("  python src/pricing_model.py")
        log.warning("  python src/demand_forecast.py")
        return {"uploaded": 0, "files": []}

    log.info(f"Uploading {len(files)} model files to s3://{bucket}/{prefix}")

    uploaded = []
    failed   = []

    for f in files:
        s3_key = f"{prefix}/{f['s3_key']}"
        size_mb = f["local"].stat().st_size / 1024 / 1024

        if dry_run:
            log.info(f"  [DRY RUN] Would upload: {f['local'].name} "
                     f"({size_mb:.1f}MB) → s3://{bucket}/{s3_key}")
            uploaded.append(f["s3_key"])
            continue

        try:
            log.info(f"  Uploading {f['local'].name} ({size_mb:.1f}MB)...")
            client.upload_file(
                str(f["local"]),
                bucket,
                s3_key,
                ExtraArgs={"Metadata": {
                    "description": f["description"],
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "size_bytes":  str(f["local"].stat().st_size),
                }}
            )
            log.info(f"  ✓ {f['description']}")
            uploaded.append(f["s3_key"])
        except Exception as e:
            log.error(f"  ✗ Failed to upload {f['local'].name}: {e}")
            failed.append({"file": f["s3_key"], "error": str(e)})

    # Upload manifest — records what's in S3 and when it was uploaded
    if not dry_run and uploaded:
        manifest = {
            "uploaded_at":  datetime.utcnow().isoformat(),
            "files":        uploaded,
            "total_files":  len(uploaded),
            "bucket":       bucket,
            "prefix":       prefix,
        }
        client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/manifest.json",
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )
        log.info(f"  ✓ Manifest updated")

    log.info(f"\nUpload complete: {len(uploaded)} files uploaded, "
             f"{len(failed)} failed")

    return {
        "uploaded": len(uploaded),
        "failed":   len(failed),
        "files":    uploaded,
        "errors":   failed,
    }


# ── Download ──────────────────────────────────────────────────────────

def download_models(force: bool = False) -> dict:
    """
    Download model files from S3 to local paths.
    Called on Railway startup before uvicorn starts.

    force=False: skip files that already exist locally (fast)
    force=True:  always download, overwriting local files

    Returns summary of what was downloaded.
    """
    client, config = get_s3_client()
    bucket  = config["bucket"]
    prefix  = config["prefix"]

    # Check manifest first — tells us what's available in S3
    try:
        response = client.get_object(
            Bucket=bucket,
            Key=f"{prefix}/manifest.json"
        )
        manifest = json.loads(response["Body"].read())
        log.info(f"S3 manifest found: {manifest['total_files']} files "
                 f"(uploaded {manifest['uploaded_at'][:10]})")
    except client.exceptions.NoSuchKey:
        log.warning("No manifest found in S3. Models have not been uploaded yet.")
        log.warning("Run locally: python src/model_store.py --upload")
        return {"downloaded": 0, "skipped": 0, "files": []}
    except Exception as e:
        log.error(f"Could not read S3 manifest: {e}")
        return {"downloaded": 0, "skipped": 0, "error": str(e)}

    # Download each file in the manifest
    downloaded = []
    skipped    = []
    failed     = []

    # Map S3 keys back to local paths
    from config.settings import get_settings
    settings = get_settings()

    key_to_local = {
        "xgboost/model.ubj":      settings.xgb_model_path,
        "xgboost/metadata.json":  settings.xgb_meta_path,
    }
    prophet_dir = settings.prophet_dir
    for category in ["electronics", "fashion", "home_goods", "sports"]:
        key_to_local[f"prophet/{category}.pkl"]    = prophet_dir / f"prophet_{category}.pkl"
        key_to_local[f"prophet/{category}_meta.json"] = prophet_dir / f"prophet_{category}_meta.json"

    for s3_key in manifest["files"]:
        local_path = key_to_local.get(s3_key)
        if not local_path:
            log.warning(f"  Unknown S3 key: {s3_key} — skipping")
            continue

        # Skip if file exists and force=False
        if local_path.exists() and not force:
            log.info(f"  [SKIP] {local_path.name} already exists")
            skipped.append(s3_key)
            continue

        full_s3_key = f"{prefix}/{s3_key}"
        try:
            # Create parent directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            log.info(f"  Downloading {local_path.name}...")
            client.download_file(bucket, full_s3_key, str(local_path))
            size_mb = local_path.stat().st_size / 1024 / 1024
            log.info(f"  ✓ {local_path.name} ({size_mb:.1f}MB)")
            downloaded.append(s3_key)
        except Exception as e:
            log.error(f"  ✗ Failed to download {s3_key}: {e}")
            failed.append({"file": s3_key, "error": str(e)})

    log.info(f"\nDownload complete: {len(downloaded)} downloaded, "
             f"{len(skipped)} skipped, {len(failed)} failed")

    return {
        "downloaded": len(downloaded),
        "skipped":    len(skipped),
        "failed":     len(failed),
        "files":      downloaded,
        "errors":     failed,
    }


# ── Status ────────────────────────────────────────────────────────────

def check_status() -> dict:
    """Show what model files exist in S3 and locally."""
    client, config = get_s3_client()
    bucket  = config["bucket"]
    prefix  = config["prefix"]

    print(f"\n{'='*55}")
    print("ARIA Model Store Status")
    print(f"  Bucket: s3://{bucket}/{prefix}")
    print(f"{'='*55}")

    # S3 files
    print("\nS3 (cloud):")
    try:
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{prefix}/"
        )
        objects = response.get("Contents", [])
        if not objects:
            print("  No files found")
        else:
            for obj in objects:
                key      = obj["Key"].replace(f"{prefix}/", "")
                size_mb  = obj["Size"] / 1024 / 1024
                modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
                print(f"  ✓ {key:<45} {size_mb:6.1f}MB  {modified}")
    except Exception as e:
        print(f"  Error: {e}")

    # Local files
    print("\nLocal:")
    files = get_model_files()
    if not files:
        print("  No model files found locally")
    else:
        for f in files:
            size_mb = f["local"].stat().st_size / 1024 / 1024
            print(f"  ✓ {f['local'].name:<45} {size_mb:6.1f}MB")

    print(f"{'='*55}\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARIA S3 Model Store")
    parser.add_argument("--upload",   action="store_true",
                        help="Upload local models to S3")
    parser.add_argument("--download", action="store_true",
                        help="Download models from S3")
    parser.add_argument("--status",   action="store_true",
                        help="Show S3 and local model status")
    parser.add_argument("--force",    action="store_true",
                        help="Force download even if files exist locally")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be uploaded without doing it")
    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.upload:
        upload_models(dry_run=args.dry_run)
    elif args.download:
        download_models(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()