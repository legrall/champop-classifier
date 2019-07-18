def test_healthcheck(client):
    '''
    GIVEN a Flask application
    WHEN a GET request to '/api/v1/healthcheck' is trigger
    THEN the application return a 200 and a valid message
    '''
    response = client.get('/api/v1/healthcheck')

    assert response.status_code == 200
