package main.kotlin.train.deepReinforcement

import main.kotlin.network.*
import main.kotlin.reinforcement.PPOAgent

class ReinforcementLearning {

    fun foodGameTraining() {
        val environment = FoodGameEnvironment()
        val model = NeuralNetwork(optimizer = Adam(0.01, 0.9, 0.999))
        model.addLayer(Layer(25, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(40, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(20, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(FoodGameAction.entries.size, Softmax, NormalXavierGlorotInitialization))

        val agent = PPOAgent(model, environment)

        agent.train(1000, 1, 0.99, 0.2)
    }
}