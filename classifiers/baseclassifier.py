class BaseClassifier:
    def __init__(self):
        self.trained = False

    def train(self, trainingset):
        pass

    def predict(self, testpoints):
        pass

    def error(self, predicted, actual):
        pass
