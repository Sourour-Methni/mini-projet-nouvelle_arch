# Importez les bibliothèques nécessaires
from flask import Flask, render_template, request, jsonify

# Créez une instance Flask
app = Flask(__name__)

# Créez une route pour la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=500)
