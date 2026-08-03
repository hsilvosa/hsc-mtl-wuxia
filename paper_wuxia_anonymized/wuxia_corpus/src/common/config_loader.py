"""
Configuration loader for corpus processing pipeline.
Centralizes all paths and parameters from config.yaml
"""

import os
import yaml
from pathlib import Path

class CorpusConfig:
    """Load and manage corpus configuration"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Try to find config.yaml in parent directories
            current = Path(__file__).parent
            while current != current.parent:
                config_file = current.parent.parent / "config.yaml"
                if config_file.exists():
                    config_path = config_file
                    break
                current = current.parent
        
        if config_path is None:
            raise FileNotFoundError("config.yaml not found")
        
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent
        self._config = self._load_config()
    
    def _load_config(self):
        """Load YAML configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_dataset_config(self, dataset_name):
        """Get configuration for a specific dataset"""
        return self._config['datasets'][dataset_name]
    
    def get_dataset_path(self, dataset_name, path_type):
        """
        Get full path for dataset directory
        
        Args:
            dataset_name: 'awe', 'condor', 'gu', 'issth'
            path_type: 'raw', 'segmented', 'processed'
        
        Returns:
            Path object
        """
        dataset_config = self.get_dataset_config(dataset_name)
        relative_path = dataset_config[f'{path_type}_dir']
        return self.base_dir / relative_path
    
    def get_alignment_config(self):
        """Return the parameters for the built-in N-M aligner."""
        return self._config['alignment']
    
    def get_script_path(self, script_type):
        """Get path to script file"""
        section = self._config.get(script_type.split('_')[0])
        if section:
            return self.base_dir / section.get('script', '')
        return None
    
    def get_logs_dir(self):
        """Get logs directory"""
        return self.base_dir / self._config['output']['logs_dir']
    
    def ensure_dirs(self, *paths):
        """Create directories if they don't exist"""
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


# Singleton instance
_config_instance = None

def get_config():
    """Get singleton config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = CorpusConfig()
    return _config_instance
