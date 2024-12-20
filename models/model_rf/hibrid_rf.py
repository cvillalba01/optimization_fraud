import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gym  # Biblioteca para crear entornos RL
from stable_baselines3 import PPO  # Algoritmo de RL basado en políticas

# Cargar y preprocesar los datos
data = pd.read_csv('../../data/PreprocessData/prediction_results_202409081602_copy.csv')
X = data.drop(columns=['rrn', 'fraud_label'])
y = data['fraud_label'].astype(int)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Definir el entorno RL
class FraudDetectionEnv(gym.Env):
    def __init__(self):
        super(FraudDetectionEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Acción: 0 = No investigar, 1 = Investigar
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)
        self.current_step = 0
        self.data = X_test
        self.labels = y_test

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        reward = 0
        done = False

        # Realizar la acción
        pred_proba = model_rf.predict_proba(self.data.iloc[self.current_step].values.reshape(1, -1))[:, 1]
        true_label = self.labels.iloc[self.current_step]

        if action == 1:  # Investigar
            if true_label == 1:  # Detectar correctamente
                reward = 1
            else:  # Falso positivo
                reward = -1
        else:  # No investigar
            if true_label == 1:  # No detectado
                reward = -1

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        return self.data.iloc[self.current_step].values, reward, done, {}

# Crear el entorno
env = FraudDetectionEnv()

# Entrenar el agente RL
model_rl = PPO('MlpPolicy', env, verbose=1)
model_rl.learn(total_timesteps=10000)

# Evaluar el agente RL
state = env.reset()
done = False
while not done:
    action, _ = model_rl.predict(state)
    state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
