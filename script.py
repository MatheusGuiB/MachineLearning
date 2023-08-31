import numpy as np
import tensorflow as tf

# Dados de exemplo
np.random.seed(0)
X = np.random.rand(1000, 1) * 10
y = 2 * X + 1 + np.random.randn(1000, 1) * 1  # Menos ruído
target = 10.5  # Valor desejado
tolerance = 0.1  # Tolerância para a proximidade do valor desejado
max_iterations = 100  # Número máximo de iterações

# Criando a rede neural com arquitetura mais complexa
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=1), loss='mean_squared_error')

# Treinando até atingir a proximidade do valor desejado ou máximo de iterações
for iteration in range(max_iterations):
    # Treina por 100 épocas a cada iteração
    model.fit(X, y, epochs=10, verbose=0)
    predicted_value = model.predict([target])[0][0]

    if abs(predicted_value - target) < tolerance:
        break

print(f'Valor Desejado: {target}')
print(f'Previsão Obtida pela Rede Neural: {predicted_value}')
