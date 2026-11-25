"""Hydra plugin auto-registering SRTrainer Structured Configs in ConfigStore."""

from __future__ import annotations

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from config.store import register_configs


class SRTrainerConfigPlugin(SearchPathPlugin):
    """Ensure SRTrainer ConfigStore schemas are always registered when Hydra initializes."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        register_configs()
        _ = search_path
