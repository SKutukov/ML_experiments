import numpy as np

class Network:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(4, input_dim=8, activation='relu'))
        self.model.add(Dense(4, input_dim=4, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, Y, epochs, batch_size):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    def predict(self, env, s):
        return self.model.predict(s)

class Table:
    def __init__(self):
        self.table = np.random.rand(4, 8)
        self.alpha = 0.05
        self.gamma = 0.5
  
    def fit(self, s_0, s_1, reward, a):
        
        self.table = self.table @ np.array(s_0)[a] + self.alpha * (reward + self.gamma *
             np.argmax(self.table @ np.array(s_1)))         

    def predict(self, env, s):
        print(s)
        return np.argmax(self.table @ np.array(s))