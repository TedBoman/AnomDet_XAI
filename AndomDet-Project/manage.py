#Manage Frontend-Backend communication
#!/usr/bin/env python3
import model_engine
from model_lstm import LSTMModel


def main():
    model_instance = LSTMModel()
    model_engine.run_lstm(model_instance)


if __name__ == '__main__':
    main()




