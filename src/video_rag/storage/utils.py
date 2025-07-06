"""
Storage Utilities for MCP Video RAG System.

This module provides utility functions and classes for storage operations
including file hashing, quota management, metrics collection, and error handling.
"""

import hashlib
import logging
import mimetypes
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import uuid
from datetime import datetime, timedelta

from .models import StorageItem, StorageUsage, StorageConfiguration


class StorageException(Exception):
    """Base exception for storage operations."""
    pass


class StorageQuotaExceeded(StorageException):
    """Exception raised when storage quota is exceeded."""
    pass


class StorageCorruption(StorageException):
    """Exception raised when storage corruption is detected."""
    pass


class FileHasher:
    """Utility class for file hashing operations."""
    
    SUPPORTED_ALGORITHMS = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
    }
    
    @classmethod
    def hash_file(
        cls, 
        file_path: Path, 
        algorithm: str = 'sha256',
        chunk_size: int = 8192
    ) -> str:
        """Calculate hash of a file."""
        if algorithm not in cls.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        hash_func = cls.SUPPORTED_ALGORITHMS[algorithm]()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            raise StorageException(f"Failed to hash file {file_path}: {e}")
    
    @classmethod
    def verify_file(
        cls, 
        file_path: Path, 
        expected_hash: str, 
        algorithm: str = 'sha256'
    ) -> bool:
        """Verify file integrity using hash."""
        try:
            actual_hash = cls.hash_file(file_path, algorithm)
            return actual_hash == expected_hash
        except Exception:
            return False


class StorageQuota:
    """Storage quota management."""
    
    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def check_quota(self, file_size: int) -> bool:
        """Check if adding a file would exceed quota."""
        try:
            current_usage = self.get_current_usage()
            
            # Check file count quota
            if current_usage.total_files >= self.config.max_files:
                return False
            
            # Check storage size quota
            if current_usage.used_space + file_size > self.config.max_storage_size:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking quota: {e}")
            return False
    
    def get_current_usage(self) -> StorageUsage:
        """Get current storage usage."""
        usage = StorageUsage()
        
        try:
            if not self.config.base_path.exists():
                return usage
            
            total_size = 0
            file_count = 0
            files_by_type = {}
            size_by_type = {}
            oldest_file = None
            newest_file = None
            
            for file_path in self.config.base_path.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    # Track by file type
                    file_type = file_path.suffix.lower() or 'unknown'
                    files_by_type[file_type] = files_by_type.get(file_type, 0) + 1
                    size_by_type[file_type] = size_by_type.get(file_type, 0) + file_size
                    
                    # Track oldest and newest files
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if oldest_file is None or file_time < oldest_file:
                        oldest_file = file_time
                    if newest_file is None or file_time > newest_file:
                        newest_file = file_time
            
            # Calculate available space
            statvfs = os.statvfs(self.config.base_path)
            available_space = statvfs.f_bavail * statvfs.f_frsize
            
            usage.total_files = file_count
            usage.total_size = self.config.max_storage_size
            usage.used_space = total_size
            usage.available_space = available_space
            usage.files_by_type = files_by_type
            usage.size_by_type = size_by_type
            usage.oldest_file = oldest_file
            usage.newest_file = newest_file
            
        except Exception as e:
            self.logger.error(f"Error calculating usage: {e}")
        
        return usage
    
    def get_available_space(self) -> int:
        """Get available space in bytes."""
        usage = self.get_current_usage()
        return self.config.max_storage_size - usage.used_space
    
    def get_utilization_percentage(self) -> float:
        """Get storage utilization percentage."""
        usage = self.get_current_usage()
        return usage.utilization_percentage()


class StorageMetrics:
    """Storage metrics collection and analysis."""
    
    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history: List[Dict] = []
    
    def collect_metrics(self) -> Dict:
        """Collect current storage metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_files': 0,
            'total_size': 0,
            'average_file_size': 0.0,
            'file_types': {},
            'storage_utilization': 0.0,
            'largest_files': [],
            'recent_files': [],
            'error_count': 0,
        }
        
        try:
            if not self.config.base_path.exists():
                return metrics
            
            files_info = []
            total_size = 0
            file_types = {}
            
            for file_path in self.config.base_path.rglob('*'):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        file_size = stat.st_size
                        file_type = file_path.suffix.lower() or 'unknown'
                        
                        files_info.append({
                            'path': str(file_path),
                            'size': file_size,
                            'type': file_type,
                            'modified': datetime.fromtimestamp(stat.st_mtime)
                        })
                        
                        total_size += file_size
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing file {file_path}: {e}")
                        metrics['error_count'] += 1
            
            metrics['total_files'] = len(files_info)
            metrics['total_size'] = total_size
            metrics['file_types'] = file_types
            
            if files_info:
                metrics['average_file_size'] = total_size / len(files_info)
                
                # Get largest files
                largest_files = sorted(files_info, key=lambda x: x['size'], reverse=True)[:10]
                metrics['largest_files'] = [
                    {'path': f['path'], 'size': f['size']} for f in largest_files
                ]
                
                # Get recent files
                recent_files = sorted(files_info, key=lambda x: x['modified'], reverse=True)[:10]
                metrics['recent_files'] = [
                    {'path': f['path'], 'modified': f['modified'].isoformat()} 
                    for f in recent_files
                ]
            
            # Calculate storage utilization
            if self.config.max_storage_size > 0:
                metrics['storage_utilization'] = (total_size / self.config.max_storage_size) * 100
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            metrics['error_count'] += 1
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 metrics
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metric for metric in self.metrics_history
            if datetime.fromisoformat(metric['timestamp']) >= cutoff_time
        ]
    
    def analyze_trends(self) -> Dict:
        """Analyze storage trends."""
        if len(self.metrics_history) < 2:
            return {}
        
        latest = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        return {
            'file_count_change': latest['total_files'] - previous['total_files'],
            'size_change': latest['total_size'] - previous['total_size'],
            'utilization_change': latest['storage_utilization'] - previous['storage_utilization'],
            'growth_rate': self._calculate_growth_rate(),
        }
    
    def _calculate_growth_rate(self) -> float:
        """Calculate storage growth rate."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        current = self.metrics_history[-1]['total_size']
        previous = self.metrics_history[-2]['total_size']
        
        if previous == 0:
            return 0.0
        
        return ((current - previous) / previous) * 100


class StorageUtils:
    """General storage utility functions."""
    
    @staticmethod
    def get_file_type(file_path: Path) -> str:
        """Get file type based on extension and MIME type."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if mime_type:
            return mime_type
        
        # Fallback to extension
        extension = file_path.suffix.lower()
        if extension:
            return f"application/{extension[1:]}"
        
        return "application/octet-stream"
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    @staticmethod
    def generate_storage_id() -> str:
        """Generate a unique storage ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def create_directory_structure(base_path: Path, storage_id: str) -> Path:
        """Create directory structure for storage ID."""
        # Create nested directories based on storage ID
        # e.g., ab/cd/ef/abcdef... for better file system performance
        id_parts = [storage_id[i:i+2] for i in range(0, min(6, len(storage_id)), 2)]
        
        storage_path = base_path
        for part in id_parts:
            storage_path = storage_path / part
        
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path
    
    @staticmethod
    def safe_file_move(source: Path, destination: Path) -> bool:
        """Safely move a file with error handling."""
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(source), str(destination))
            return True
        except Exception as e:
            logging.error(f"Error moving file {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def safe_file_copy(source: Path, destination: Path) -> bool:
        """Safely copy a file with error handling."""
        try:
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(str(source), str(destination))
            return True
        except Exception as e:
            logging.error(f"Error copying file {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def safe_file_delete(file_path: Path) -> bool:
        """Safely delete a file with error handling."""
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
            return False
    
    @staticmethod
    def calculate_directory_size(directory: Path) -> int:
        """Calculate total size of a directory."""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logging.error(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    @staticmethod
    def find_duplicate_files(directory: Path) -> Dict[str, List[Path]]:
        """Find duplicate files in a directory based on hash."""
        duplicates = {}
        file_hashes = {}
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    try:
                        file_hash = FileHasher.hash_file(file_path)
                        if file_hash in file_hashes:
                            if file_hash not in duplicates:
                                duplicates[file_hash] = [file_hashes[file_hash]]
                            duplicates[file_hash].append(file_path)
                        else:
                            file_hashes[file_hash] = file_path
                    except Exception as e:
                        logging.warning(f"Error hashing file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error finding duplicates in {directory}: {e}")
        
        return duplicates
    
    @staticmethod
    def cleanup_empty_directories(directory: Path) -> int:
        """Remove empty directories recursively."""
        removed_count = 0
        
        try:
            for dir_path in sorted(directory.rglob('*'), reverse=True):
                if dir_path.is_dir() and dir_path != directory:
                    try:
                        dir_path.rmdir()  # Only removes if empty
                        removed_count += 1
                    except OSError:
                        # Directory not empty, continue
                        pass
        except Exception as e:
            logging.error(f"Error cleaning up empty directories in {directory}: {e}")
        
        return removed_count 