import os
import pytest
from app import app, model

def test_flask_app():
    # Test that Flask client starts successfully
    tester = app.test_client()
    response = tester.get('/')
    assert response.status_code == 200

def test_model_loaded():
    # Test that the CNN model is successfully loaded in memory
    assert model is not None