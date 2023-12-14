pipeline {
    agent any
    
    stages {
        stage('Clonage du code') {
            steps {
                git branch: 'main', url: 'https://github.com/Sourour-Methni/projet-nouvelle_arch.git'
            }
        }
        
        stage('Installation des dépendances') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Tests unitaires') {
            steps {
                sh 'python -m unittest tests/test_flask_services.py'
            }
            post {
                always {
                    junit 'tests/results/unit_tests.xml'
                }
            }
        }
    }
    
    post {
        success {
            echo 'Les tests ont réussi !'
        }
        failure {
            echo 'Les tests ont échoué !'
        }
    }
}
