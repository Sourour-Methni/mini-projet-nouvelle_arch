import requests
import unittest

class TestIntegration(unittest.TestCase):

    def setUp(self):
        # Configurez votre environnement de tests pour les requêtes HTTP
        self.base_url = 'http://localhost:510'  

    def test_svm_service_integration(self):
        data = {
            'wav_music': '/app/audio1_base64.txt'
        }
        response = requests.post(f'{self.base_url}/svm_service', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('genre', response.json())  # Vérifiez la présence de la clé 'genre' dans la réponse

    def test_predict_genre(self):

        data = {
            'wav_music': '/app/audio1_base64.txt'
        }
        response = self.app.post('/svm_service', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('genre', response.json())
        # Assurez-vous que la réponse est celle attendue
        expected_genre = 'rock'  # Remplacez par le genre attendu pour vos données de test
        self.assertEqual(response.json()['genre'], expected_genre)
