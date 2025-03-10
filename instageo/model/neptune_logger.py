"""Neptune Logger Module modified from Lyra https://github.com/instadeepai/lyra/blob/main/lyra/neptune_logger.py."""  # noqa: E501
import logging
import os
from typing import Any

import neptune
from pytorch_lightning.loggers import NeptuneLogger


def aichor_log_git_info(run: neptune.Run) -> None:
    """Manually log git information when creating a Neptune run on AIchor.

    https://gitlab.com/instadeep/deepbio/-/blob/4755d8a1006cb584edaed42d8df48f94c4b0d636/deepbio/utils/neptune_utils.py#L61 # noqa: E501

    Since the .git directory is not uploaded to AIchor instances, Neptune cannot automatically
    extract git data. Instead, AIchor exposes the following environment variables:

    - VCS_AUTHOR_EMAIL: commit author email
    - VCS_AUTHOR_NAME: commit author username
    - VCS_REF_NAME: the branch name
    - VCS_REPOSITORY: name of the repository
    - VCS_SHA: hash of the current commit
    - VCS_SHA_SHORT: hash (short) of the current commit
    - VCS_COMMIT_MESSAGE: message associated with commit
    - VCS_COMMIT_MESSAGE_SHORT: short message associated with commit
    - VCS_TYPE: remote platform (e.g. "gitlab")

    These are logged under the 'git/' namespace.
    """
    run["git/commit_author"].append(os.environ["VCS_AUTHOR_NAME"])
    run["git/commit_author_email"].append(os.environ["VCS_AUTHOR_EMAIL"])
    run["git/branch_name"].append(os.environ["VCS_REF_NAME"])
    run["git/repository"].append(os.environ["VCS_REPOSITORY"])
    run["git/commit_hash"].append(os.environ["VCS_SHA"])
    run["git/commit_hash_short"].append(os.environ["VCS_SHA_SHORT"])
    run["git/commit_message"].append(os.environ["VCS_COMMIT_MESSAGE"])
    run["git/commit_message_short"].append(os.environ["VCS_COMMIT_MESSAGE_SHORT"])

    # We "replace" rather than append `git/last_commit_url` since append mode stops the URL working
    # in the Neptune UI
    if os.environ["VCS_TYPE"] == "gitlab":
        run[
            "git/last_commit_url"
        ] = f"https://gitlab.com/{os.environ['VCS_REPOSITORY']}/-/commit/{os.environ['VCS_SHA']}"
    elif os.environ["VCS_TYPE"] == "github":
        run[
            "git/last_commit_url"
        ] = f"https://github.com/{os.environ['VCS_REPOSITORY']}/commit/{os.environ['VCS_SHA']}"


class AIchorNeptuneLogger(NeptuneLogger):
    """Custom NeptuneLogger for AIChor, inheriting from the PyTorch Lightning NeptuneLogger."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Extend the base NeptuneLogger initialization while passing args and kwargs.

        See https://gitlab.com/instadeep/biondeep/-/blob/52c1189eaf92a903dc14dc77b0aa15482fc2af17/biondeep/utils/neptune.py#L47.  # noqa: E501
        """
        super().__init__(*args, **kwargs)

        if "VCS_AUTHOR_EMAIL" in os.environ:
            aichor_log_git_info(self.run)


def set_neptune_api_token() -> str | None:
    """Sets the NEPTUNE_API_TOKEN based on the VCS author's email, if available.

    Returns:
        str | None: The extracted Neptune API token if found, otherwise None.
    """
    if "VCS_AUTHOR_EMAIL" in os.environ:
        author_email = os.environ["VCS_AUTHOR_EMAIL"]
        author_username, _ = author_email.split("@")
        author_username = author_username.replace("-", "_").replace(".", "_").upper()
        env_var_name = f"{author_username}__NEPTUNE_API_TOKEN"

        if env_var_name in os.environ:
            author_api_token = os.environ[env_var_name]
            os.environ["NEPTUNE_API_TOKEN"] = author_api_token
            return author_api_token
        else:
            logging.warning(
                f"Neptune credentials for user {author_username} not found. "
                "Using environment variable NEPTUNE_API_TOKEN instead."
            )

    return None
