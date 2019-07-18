import shutil
from pathlib import Path

import pytest
from classifier import create_app

@pytest.fixture
def app():
    app = create_app({
        'TESTING': True
    })

    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


@pytest.fixture(scope='module')
def input_audio_path():
    input_file = Path(__file__).parent / 'fixtures' / 'test_1.wav'
    return input_file.resolve()


@pytest.fixture(scope='module')
def input_audio_file(input_audio_path):
    input_file = open(input_audio_path, mode='r')
    yield input_file
    input_file.close()


@pytest.fixture(scope='module')
def output_tmp_folder():
    output_folder = Path('/tmp') / 'converter_tests'

    assert not output_folder.exists()
    output_folder.mkdir(parents=True)

    yield output_folder

    if output_folder.exists():
        shutil.rmtree(output_folder)
