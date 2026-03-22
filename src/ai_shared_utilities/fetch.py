from pathlib import Path
from dataclasses import replace

from ai_shared_utilities.assets import get_asset_home
from ai_shared_utilities.registry import ASSETS, REGISTRY, Asset


def get_asset(name: str) -> Asset:
    asset = REGISTRY[name]
    resolved_path = get_asset_home(asset.kind) / asset.relative_path
    return replace(asset, path=resolved_path)


def asset_exists(name: str) -> bool:
    try:
        path = get_asset_path(name)
    except KeyError:
        return False
    return path.exists()


def get_asset_path(name: str) -> Path:
    """
    Return the full filesystem path for an asset.

    Raises KeyError if the asset name is not registered.
    """
    asset = ASSETS[name]
    return get_asset_home(asset.kind) / asset.relative_path


def ensure_asset(name: str, rebuild: bool = False) -> Path:
    """
    Ensure the asset exists locally.

    Returns the asset path if found.
    Raises FileNotFoundError if the asset is missing.
    """
    asset = ASSETS[name]
    path = get_asset_path(name)

    if path.exists() and not rebuild:
        return path

    if asset.builder:
        print(f"Building asset: {name}")
        asset.builder()

    if not path.exists():
        raise FileNotFoundError(
            f"Asset '{name}' not found at {path}"
        )

    return path
