import random

from flask import jsonify
from flask_restful import Resource

from .utils import classify_hand_player


class HealthCheck(Resource):
    def get(self):
        return {'status' : 'Alive babe'}

class Play(Resource):
    def post(self):
        n = random.randrange(10)
        if n <= 2:
            return jsonify({
                'play': 'raise'
            })
        if n <= 5:
            return jsonify({
                'play': 'call'
            })
        else:
            return jsonify({
                'play': 'fold'
            })
