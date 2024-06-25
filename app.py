from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el scaler ajustado
model = joblib.load('modelo.joblib')
#scaler = joblib.load('scaler.joblib')# esto sirve para escalar  los datos de las cajas de texto ver proyecto de fin de mes 1
app.logger.debug('Modelo y scaler cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        long = float(request.form['longPico'])
        pro = float(request.form['proPico'])
        masa = float(request.form['masa'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[long, pro, masa]], columns=['bill_length_mm', 'bill_depth_mm', 'body_mass_g'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Escalar los datos utilizando el scaler ajustado anteriormente,se deben de escalar los datos mandados ya que en el standar scales
        # con el que hicimos el entrenamiendo del arbol de desicion esta entrenado con numeros escalados
        scalerX = scaler.transform(data_df)
        scaler_df = pd.DataFrame(scalerX, columns=data_df.columns)
        app.logger.debug(f'DataFrame escalado: {scaler_df}')
        
        # Realizar predicciones
        prediction = model.predict(scaler_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a un tipo de datos serializable (int)
        prediction_serializable = int(prediction[0])
        
        if prediction_serializable == 0:
            dato = "Adelie"
        elif prediction_serializable == 1:
            dato = "Chinstrap"
        elif prediction_serializable == 2:
            dato = "Gentoo"
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': dato})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=port)