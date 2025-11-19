"""
Cloud storage sync for benchmark results using S3-compatible storage
"""

import logging
import os
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


class CloudSyncManager:
    """Manager for syncing benchmark results with S3-compatible cloud storage."""

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        prefix: str = "",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ):
        """Initialize CloudSyncManager.

        Args:
            endpoint_url: S3-compatible endpoint URL
            bucket: Bucket name
            prefix: Optional prefix for all keys (e.g., "benchmark-results/")
            aws_access_key_id: Access key (defaults to env var)
            aws_secret_access_key: Secret key (defaults to env var)
        """
        if boto3 is None:
            raise ImportError("boto3 is required for cloud sync. Install with: uv add boto3")

        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def list_remote_files(self, run_id: str) -> set[str]:
        """List all files for a run in cloud storage.

        Args:
            run_id: Run directory name

        Returns:
            Set of relative file paths within the run
        """
        prefix = f"{self.prefix}{run_id}/"
        files = set()
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue
                    
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    # Remove prefix and run_id to get relative path
                    rel_path = s3_key[len(prefix):]
                    if rel_path:  # Skip directory markers
                        files.add(rel_path)
        except ClientError as e:
            logger.error(f"Failed to list remote files for {run_id}: {e}")
            
        return files

    def push_run(self, run_dir: str, progress_callback=None, skip_existing=True) -> tuple[bool, int, int]:
        """Upload a single run directory to cloud storage.

        Args:
            run_dir: Path to run directory (e.g., "3667_1P_1D_20251110_192145")
            progress_callback: Optional callback function(current, total, filename, status)
            skip_existing: If True, skip files that already exist in cloud

        Returns:
            Tuple of (success: bool, uploaded: int, skipped: int)
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            logger.error(f"Run directory does not exist: {run_dir}")
            return False, 0, 0

        run_name = run_path.name
        logger.info(f"Pushing run {run_name} to cloud storage...")

        # Get all local files
        local_files = []
        for root, _dirs, files in os.walk(run_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(run_path)
                local_files.append((file_path, str(rel_path)))

        if not local_files:
            logger.warning(f"No files found in {run_dir}")
            return False, 0, 0

        # Get remote files if we're skipping existing
        remote_files = set()
        if skip_existing:
            logger.info(f"Checking remote files for {run_name}...")
            remote_files = self.list_remote_files(run_name)
            logger.info(f"Found {len(remote_files)} files already in cloud")

        # Upload missing files
        uploaded = 0
        skipped = 0
        total = len(local_files)
        
        for file_path, rel_path in local_files:
            s3_key = f"{self.prefix}{run_name}/{rel_path}"
            
            # Check if file already exists remotely
            if skip_existing and rel_path in remote_files:
                skipped += 1
                if progress_callback:
                    progress_callback(uploaded + skipped, total, rel_path, "skipped")
                logger.debug(f"Skipped {rel_path} (already in cloud)")
                continue

            try:
                self.s3.upload_file(str(file_path), self.bucket, s3_key)
                uploaded += 1

                if progress_callback:
                    progress_callback(uploaded + skipped, total, rel_path, "uploaded")

                logger.debug(f"Uploaded {rel_path}")
            except Exception as e:
                logger.error(f"Failed to upload {rel_path}: {e}")
                return False, uploaded, skipped

        logger.info(f"Successfully pushed {uploaded} files from {run_name} ({skipped} skipped)")
        return True, uploaded, skipped

    def pull_run(self, run_id: str, local_dir: str, progress_callback=None, skip_existing=True) -> tuple[str | None, int, int]:
        """Download a single run from cloud storage.

        Args:
            run_id: Run directory name (e.g., "3667_1P_1D_20251110_192145")
            local_dir: Local directory to download to
            progress_callback: Optional callback function(current, total, filename, status)
            skip_existing: If True, skip files that already exist locally

        Returns:
            Tuple of (path: str | None, downloaded: int, skipped: int)
        """
        logger.info(f"Pulling run {run_id} from cloud storage...")

        # List all remote files
        remote_files = self.list_remote_files(run_id)
        if not remote_files:
            logger.warning(f"No files found for run {run_id}")
            return None, 0, 0

        run_path = Path(local_dir) / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Determine which files to download
        files_to_download = []
        skipped = 0
        
        for rel_path in remote_files:
            local_file = run_path / rel_path
            
            # Skip if exists locally
            if skip_existing and local_file.exists():
                skipped += 1
                if progress_callback:
                    progress_callback(len(files_to_download) + skipped, len(remote_files), rel_path, "skipped")
                logger.debug(f"Skipped {rel_path} (already exists locally)")
                continue

            files_to_download.append(rel_path)

        if not files_to_download and skipped > 0:
            logger.info(f"All {skipped} files already exist locally")
            return str(run_path), 0, skipped

        # Download missing files
        downloaded = 0
        total = len(remote_files)
        
        for rel_path in files_to_download:
            s3_key = f"{self.prefix}{run_id}/{rel_path}"
            local_file = run_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.s3.download_file(self.bucket, s3_key, str(local_file))
                downloaded += 1

                if progress_callback:
                    progress_callback(downloaded + skipped, total, rel_path, "downloaded")

                logger.debug(f"Downloaded {rel_path}")
            except Exception as e:
                logger.error(f"Failed to download {rel_path}: {e}")
                return None, downloaded, skipped

        logger.info(f"Successfully pulled {downloaded} files to {run_path} ({skipped} skipped)")
        return str(run_path), downloaded, skipped

    def list_remote_runs(self) -> list[str]:
        """List all runs available in cloud storage.

        Returns:
            List of run directory names
        """
        try:
            # Use delimiter to get "directories" at the prefix level
            response = self.s3.list_objects_v2(
                Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"
            )
        except ClientError as e:
            logger.error(f"Failed to list remote runs: {e}")
            return []

        if "CommonPrefixes" not in response:
            return []

        # Extract run names from common prefixes
        runs = []
        for prefix_obj in response["CommonPrefixes"]:
            prefix_str = prefix_obj["Prefix"]
            # Remove the base prefix and trailing slash
            run_name = prefix_str[len(self.prefix) :].rstrip("/")
            if run_name:
                runs.append(run_name)

        return sorted(runs, reverse=True)

    def sync_missing_runs(self, local_dir: str, progress_callback=None) -> tuple[int, int, int]:
        """Download missing files from cloud storage.

        This will:
        1. Check all remote runs
        2. For each run, download only files that don't exist locally

        Args:
            local_dir: Local logs directory
            progress_callback: Optional callback function(run_name, current, total, status)

        Returns:
            Tuple of (runs_synced: int, files_downloaded: int, files_skipped: int)
        """
        # Get list of remote runs
        remote_runs = self.list_remote_runs()
        if not remote_runs:
            logger.info("No remote runs found")
            return 0, 0, 0

        logger.info(f"Found {len(remote_runs)} runs in cloud storage")

        # Sync all runs (downloading only missing files)
        runs_synced = 0
        total_downloaded = 0
        total_skipped = 0
        
        for i, run_id in enumerate(remote_runs, 1):
            if progress_callback:
                progress_callback(run_id, i, len(remote_runs), "syncing")

            result_path, downloaded, skipped = self.pull_run(
                run_id, local_dir, skip_existing=True
            )
            
            if result_path:
                runs_synced += 1
                total_downloaded += downloaded
                total_skipped += skipped

        logger.info(
            f"Synced {runs_synced}/{len(remote_runs)} runs: "
            f"{total_downloaded} files downloaded, {total_skipped} files skipped"
        )
        return runs_synced, total_downloaded, total_skipped

    def run_exists_in_cloud(self, run_id: str) -> bool:
        """Check if a run exists in cloud storage.

        Args:
            run_id: Run directory name

        Returns:
            True if run exists, False otherwise
        """
        prefix = f"{self.prefix}{run_id}/"
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
            return "Contents" in response and len(response["Contents"]) > 0
        except ClientError as e:
            logger.error(f"Failed to check if run exists: {e}")
            return False

    def delete_run(self, run_id: str, progress_callback=None) -> tuple[bool, int]:
        """Delete a run from cloud storage.

        Args:
            run_id: Run directory name to delete
            progress_callback: Optional callback function(current, total, filename)

        Returns:
            Tuple of (success: bool, deleted_count: int)
        """
        logger.warning(f"Deleting run {run_id} from cloud storage...")

        # List all files for this run
        remote_files = self.list_remote_files(run_id)
        
        if not remote_files:
            logger.warning(f"No files found for run {run_id}")
            return False, 0

        logger.info(f"Found {len(remote_files)} files to delete")

        # Delete all files
        deleted = 0
        total = len(remote_files)
        
        for rel_path in remote_files:
            s3_key = f"{self.prefix}{run_id}/{rel_path}"
            
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
                deleted += 1
                
                if progress_callback:
                    progress_callback(deleted, total, rel_path)
                
                logger.debug(f"Deleted {rel_path}")
            except Exception as e:
                logger.error(f"Failed to delete {rel_path}: {e}")
                return False, deleted

        logger.info(f"Successfully deleted {deleted} files for run {run_id}")
        return True, deleted

    def test_connection(self) -> bool:
        """Test connection to cloud storage.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            logger.info("Successfully connected to cloud storage")
            return True
        except NoCredentialsError:
            logger.error("No credentials provided for cloud storage")
            return False
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                logger.error(f"Bucket '{self.bucket}' does not exist")
            else:
                logger.error(f"Failed to connect to cloud storage: {e}")
            return False


def load_cloud_config(config_path: str = "srtslurm.yaml") -> dict | None:
    """Load cloud configuration from YAML file.

    Args:
        config_path: Path to srtslurm.yaml

    Returns:
        Dict with cloud config, or None if file doesn't exist
    """
    import yaml
    
    config_file = Path(config_path)
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config.get("cloud", {}) if config else {}
    except Exception as e:
        logger.error(f"Failed to load cloud config: {e}")
        return None


def create_sync_manager_from_config(
    config_path: str = "srtslurm.yaml",
) -> CloudSyncManager | None:
    """Create CloudSyncManager from config file.

    Credentials can be provided via:
    1. YAML config (aws_access_key_id, aws_secret_access_key)
    2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    3. AWS credentials file (~/.aws/credentials)

    Args:
        config_path: Path to srtslurm.yaml

    Returns:
        CloudSyncManager instance, or None if config doesn't exist or is invalid
    """
    config = load_cloud_config(config_path)
    if not config:
        return None

    required_keys = ["endpoint_url", "bucket"]
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return None

    try:
        return CloudSyncManager(
            endpoint_url=config["endpoint_url"],
            bucket=config["bucket"],
            prefix=config.get("prefix", ""),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
        )
    except Exception as e:
        logger.error(f"Failed to create sync manager: {e}")
        return None
