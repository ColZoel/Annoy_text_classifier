import os
import time
import numpy as np
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Import the classes and functions from your module.
# Adjust the import if your module filename is different.
from modules import (
    timer,
    ModelDownloader,
    TransformerLoader,
    Encoder,
    Classifier,
)


# --- Helpers for testing ---

class DummyResponse:
    """A dummy response object to simulate requests responses."""

    def __init__(self, status_code, headers, content=b"dummy content"):
        self.status_code = status_code
        self.headers = headers
        self._content = content

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"Status code: {self.status_code}")

    def iter_content(self, chunk_size=8192):
        # Simulate streaming content by yielding the content once.
        yield self._content


# A dummy model to use with the Encoder for embedding tests.
class DummyModel:
    def encode(self, data, convert_to_numpy=True, show_progress_bar=True):
        # Returns an array where each embedding is the length of the string (as a float)
        return np.array([[float(len(text))] for text in data])


# --- Tests for timer decorator ---

def test_timer_decorator(capsys):
    """Test that the timer decorator prints timing info and returns correct output."""

    @timer
    def dummy_func(data):
        time.sleep(0.1)
        return sum(data)

    result = dummy_func([1, 2, 3])
    captured = capsys.readouterr().out
    assert "finished in" in captured
    assert result == 6


# --- Tests for ModelDownloader ---

def test_check_compression_valid_mime(monkeypatch):
    """Test that check_compression returns True for a valid MIME type (e.g. zip)."""

    def fake_head(url, allow_redirects=True):
        return DummyResponse(200, {'Content-Type': 'application/zip'})

    monkeypatch.setattr("module_under_test.requests.head", fake_head)

    downloader = ModelDownloader("http://example.com/file.zip", "/dummy")
    assert downloader.check_compression() is True


def test_check_compression_valid_extension(monkeypatch):
    """Test that check_compression works by checking the file extension (.tar)."""

    def fake_head(url, allow_redirects=True):
        return DummyResponse(200, {'Content-Type': 'application/octet-stream'})

    monkeypatch.setattr("module_under_test.requests.head", fake_head)

    downloader = ModelDownloader("http://example.com/file.tar", "/dummy")
    assert downloader.check_compression() is True


def test_check_compression_directory(monkeypatch):
    """Test that a URL pointing to a directory raises a ValueError."""

    def fake_head(url, allow_redirects=True):
        return DummyResponse(200, {'Content-Type': 'application/octet-stream'})

    monkeypatch.setattr("module_under_test.requests.head", fake_head)

    downloader = ModelDownloader("http://example.com/directory/", "/dummy")
    with pytest.raises(ValueError, match="url points to a directory"):
        downloader.check_compression()


def test_check_compression_invalid(monkeypatch):
    """Test that an unrecognized compressed file URL raises a ValueError."""

    def fake_head(url, allow_redirects=True):
        return DummyResponse(200, {'Content-Type': 'text/plain'})

    monkeypatch.setattr("module_under_test.requests.head", fake_head)

    downloader = ModelDownloader("http://example.com/file.txt", "/dummy")
    with pytest.raises(ValueError, match="The URL does not point to a recognized compressed file"):
        downloader.check_compression()


# --- Tests for TransformerLoader ---

def test_transformer_loader_from_remote(monkeypatch, tmp_path):
    """
    Test that from_remote downloads and saves the file.
    A temporary directory is used to simulate the destination.
    """
    temp_dir = tmp_path / "models"
    temp_dir.mkdir()

    # Fake HEAD response (for compression check)
    def fake_head(url, allow_redirects=True):
        return DummyResponse(200, {'Content-Type': 'application/zip'})

    monkeypatch.setattr("module_under_test.requests.head", fake_head)

    # Fake GET response (for file download)
    def fake_get(url, stream=True):
        return DummyResponse(200, {'Content-Type': 'application/zip'}, content=b"fake model data")

    monkeypatch.setattr("module_under_test.requests.get", fake_get)

    url = "http://example.com/model.zip"
    loader = TransformerLoader("dummy_model")
    downloaded_path = loader.from_remote(url, dst=str(temp_dir))

    # Check that the file exists and its content matches the fake content.
    assert os.path.exists(downloaded_path)
    with open(downloaded_path, 'rb') as f:
        content = f.read()
    assert content == b"fake model data"


def test_transformer_loader_search_local(monkeypatch, tmp_path):
    """
    Test the search_local method by monkeypatching local_models and creating a dummy file.
    """
    temp_dir = tmp_path / "dummy_models"
    temp_dir.mkdir()
    model_file = temp_dir / "dummy_model"
    model_file.write_text("dummy content")

    # Monkeypatch the local_models function (which is imported from tools.models1 in your module)
    monkeypatch.setattr("module_under_test.local_models", lambda directory: ["dummy_model", "other_model"])

    loader = TransformerLoader("dummy_model")
    found_path = loader.search_local(directory=str(temp_dir))
    assert os.path.basename(found_path) == "dummy_model"

    # Test that searching for a non-existent model raises an error.
    loader_nonexistent = TransformerLoader("nonexistent_model")
    with pytest.raises(ValueError):
        loader_nonexistent.search_local(directory=str(temp_dir))


# --- Tests for Encoder ---

def test_encoder_embed_nonbatched():
    """Test the embed method (non-batched) using a dummy model."""
    dummy_model = DummyModel()
    encoder = Encoder(dummy_model)
    data = ["hello", "world"]
    encoder.embed(np.array(data), batched=False)
    # "hello" and "world" both have length 5
    expected = np.array([[5.0], [5.0]])
    np.testing.assert_array_equal(encoder.embeds, expected)


def test_encoder_embed_batched():
    """Test the embed method (batched) using a dummy model."""
    dummy_model = DummyModel()
    encoder = Encoder(dummy_model)
    data = ["a", "ab", "abc", "abcd"]
    encoder.embed(np.array(data), batched=True, batch_size=2)
    expected = np.array([[1.0], [2.0], [3.0], [4.0]])
    np.testing.assert_array_equal(encoder.embeds, expected)


def test_encoder_save_load(tmp_path):
    """
    Test the save and load methods.
    Note: The current save implementation uses unpacking (i.e. **self.embeds) so we set embeds as a dictionary.
    """
    dummy_model = DummyModel()
    encoder = Encoder(dummy_model)
    test_array = np.array([1, 2, 3])
    encoder.embeds = {"embeds": test_array}

    file_path = tmp_path / "embeds.npz"
    encoder.save(str(file_path))

    loaded = np.load(str(file_path))
    np.testing.assert_array_equal(loaded["embeds"], test_array)


# --- Tests for Classifier ---

def test_classifier_build_predict(tmp_path):
    """Test building the Annoy index and predicting labels."""
    classifier = Classifier()
    embeddings = np.array([[0, 0], [10, 10]])
    classifier.build_tree(embeddings, num_trees=5)

    # Check that y_labels is set correctly
    np.testing.assert_array_equal(classifier.y_labels, np.arange(embeddings.shape[0]))

    # Predict using the same embeddings and labels.
    labels = np.array(["a", "b"])
    save_path = str(tmp_path / "pred.npz")
    classifier.predict(labels, embeddings, neighbors=1, save_path=save_path)

    # For each embedding the nearest neighbor should be itself.
    assert classifier.pred == list(labels)

    # Check that the file was created.
    assert os.path.exists(save_path)


def test_classifier_evaluate(tmp_path):
    """Test that evaluate creates a CSV classification report."""
    classifier = Classifier()
    y_true = np.array(["a", "b", "a"])
    y_pred = np.array(["a", "b", "b"])
    report_path = str(tmp_path / "report")
    classifier.evaluate(y_true, y_pred, save_path=report_path)

    report_file = report_path + ".csv"
    assert os.path.exists(report_file)

    report_df = pd.read_csv(report_file)
    # Check that expected columns are in the report.
    assert "label" in report_df.columns
    assert "yhat" in report_df.columns


def test_classifier_visualize(monkeypatch, tmp_path):
    """
    Test the visualize method.
    plt.show is monkeypatched to avoid blocking and a temporary file is used for saving.
    """
    classifier = Classifier()
    # Prepare dummy embeddings and labels for visualization.
    classifier.x_embeddings = [np.array([0, 0]), np.array([1, 1])]
    classifier.y_embeddings = np.array([[0, 0], [1, 1]])
    classifier.y_labels = np.array(["a", "b"])
    classifier.pred = ["a", "b"]

    monkeypatch.setattr(plt, "show", lambda: None)
    save_path = str(tmp_path / "figure.png")
    classifier.visualize(top=2, label_points=False, figsize=(5, 5), save=save_path)

    # Check that the figure file was created.
    assert os.path.exists(save_path)