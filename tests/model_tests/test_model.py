from unittest.mock import patch

from instageo.model.old_model import download_file


def test_download_file_success(tmp_path):
    test_filename = tmp_path / "test_file.txt"

    with patch("instageo.model.model.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"dummy content"
        download_file("http://example.com", str(test_filename))

        assert test_filename.exists()


def test_download_file_already_exists(tmp_path):
    test_filename = tmp_path / "test_file.txt"

    test_filename.write_text("dummy content")

    with patch("instageo.model.model.requests.get") as mock_get:
        download_file("http://example.com", str(test_filename))
        mock_get.assert_not_called()
