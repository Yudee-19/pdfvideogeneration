import os
import logging
import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError
from boto3.s3.transfer import TransferConfig
from app.config import settings
import json
from typing import Optional
logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.bucket_name = settings.AWS_BUCKET_NAME
        self.region = settings.AWS_REGION
        
        # Initialize only if credentials exist
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            # Config for large files (3GB+): Multipart upload chunks
            self.transfer_config = TransferConfig(
                multipart_threshold=1024 * 25, # 25MB threshold
                max_concurrency=10,
                multipart_chunksize=1024 * 25,
                use_threads=True
            )
            logger.info(f"S3 Manager Initialized (Bucket: {self.bucket_name})")
        else:
            self.s3 = None
            logger.warning("AWS Credentials missing. S3 features disabled.")

    def upload_file(self, local_path: Path, s3_key: str):
        """Uploads a single file to S3 efficiently."""
        if not self.s3: return False
        
        try:
            logger.info(f"Uploading {local_path.name} to S3...")
            self.s3.upload_file(
                str(local_path), 
                self.bucket_name, 
                s3_key,
                Config=self.transfer_config
            )
            return True
        except Exception as e:
            logger.error(f"Upload failed for {local_path}: {e}")
            return False

    def download_file(self, s3_key: str, local_path: Path):
        """Downloads a single file from S3."""
        if not self.s3: return False
        
        try:
            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {s3_key} from S3...")
            self.s3.download_file(
                self.bucket_name, 
                s3_key, 
                str(local_path),
                Config=self.transfer_config
            )
            return True
        except Exception as e:
            logger.error(f"Download failed for {s3_key}: {e}")
            return False

    def sync_job_to_s3(self, job_id: str):
        """
        Replicates the entire local job folder to S3.
        Useful for backing up large audio/video files.
        """
        if not self.s3: return
        
        job_dir = settings.JOBS_OUTPUT_PATH / job_id
        if not job_dir.exists():
            logger.warning(f"Job dir {job_dir} does not exist, skipping S3 sync.")
            return

        logger.info(f"Syncing job {job_id} to S3...")
        uploaded_count = 0
        
        for root, _, files in os.walk(job_dir):
            for file in files:
                local_path = Path(root) / file
                # Create S3 Key: jobs/{job_id}/filename.mp4
                relative_path = local_path.relative_to(settings.JOBS_OUTPUT_PATH)
                s3_key = f"jobs/{relative_path}".replace("\\", "/") # Fix for Windows paths
                
                if self.upload_file(local_path, s3_key):
                    uploaded_count += 1
                    
        logger.info(f"Synced {uploaded_count} files for job {job_id} to S3.")

    def sync_job_from_s3(self, job_id: str):
        """
        Downloads the entire job folder from S3 to local.
        Useful when a new worker needs to pick up an existing job.
        """
        if not self.s3: return

        prefix = f"jobs/{job_id}/"
        try:
            # List all objects in the job folder
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            for page in pages:
                if 'Contents' not in page: continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Calculate local path
                    relative_path = s3_key.replace(f"jobs/", "", 1)
                    local_path = settings.JOBS_OUTPUT_PATH / relative_path
                    
                    self.download_file(s3_key, local_path)
                    
            logger.info(f"Job {job_id} successfully synced from S3 to local.")
            
        except Exception as e:
            logger.error(f"Failed to sync job {job_id} from S3: {e}")

    def generate_presigned_url(self, s3_key: str, expiration=3600):
        """Generate a temporary URL for direct download from S3."""
        if not self.s3: return None
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None
        
    def get_job_metadata_from_s3(self, job_id: str) -> Optional[dict]:
        """
        Fetches job metadata directly from S3 memory.
        Used when local files have been cleaned up.
        """
        if not self.s3: return None
        
        s3_key = f"jobs/{job_id}/job_metadata.json"
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_content = response['Body'].read().decode('utf-8')
            return json.loads(json_content)
        except Exception as e:
            # It's normal to fail if the job hasn't uploaded metadata yet
            return None


# Global instance
s3_manager = S3Manager()