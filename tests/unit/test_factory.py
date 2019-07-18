from classifier import create_app

def test_flask_factory(app):
    assert not create_app().testing
    assert create_app({'TESTING': True}).testing
