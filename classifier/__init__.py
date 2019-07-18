from pathlib import Path

from flask import Flask, request, send_file
from flask_cors import CORS
from flask_restful import Resource, Api

from .routes import HealthCheck, Play


def create_app(test_config=None):
    '''Create and configure an instance of the Flask'''
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        UPLOAD_FOLDER='/tmp/images',
        MAX_CONTENT_LENGTH=30 * 1024 * 1024
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # Uploads Folder setup 
    Path(f"{app.config['UPLOAD_FOLDER']}").mkdir(parents=True, exist_ok=True)

    api = Api(app, prefix='/api/v1')
    api.add_resource(HealthCheck, '/healthcheck')
    api.add_resource(Play, '/gameplay/preflop/play')

    return app
