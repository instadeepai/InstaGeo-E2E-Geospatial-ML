# Uncomment the below to simulate multi-device Jax on your local machine.
# import os
#
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import logging

import hydra
import jax
from ml_research_template.module import MyClass
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Uncomment the below to disable jit (for development purposes only).
# from jax.config import config
# config.update("jax_disable_jit", True)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="default",
)
def run(cfg: DictConfig) -> None:
    """Sample run function."""
    log.info(f"Script: {__file__}")
    log.info(f"Imported hydra config:\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Imported module: MyClass.some_variable: {MyClass.some_variable}")
    log.info(f"[JAX] Local devices: {jax.local_devices()}.")
    log.info(f"[JAX] Global devices: {jax.devices()}.")
    log.info("Demo script finished!")


if __name__ == "__main__":
    run()
