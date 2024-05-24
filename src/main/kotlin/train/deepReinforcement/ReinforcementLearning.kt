package main.kotlin.train.deepReinforcement

import main.kotlin.network.*

class ReinforcementLearning {

    fun foodGameTraining() {
        val environment = FoodGameEnvironment()
        val model = NeuralNetwork(
            trainingMethod = PPOTraining(environment),
            optimizer = Adam(0.01, 0.9, 0.999)
        )
        model.addLayer(Layer(25, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(40, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(20, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(FoodGameAction.entries.size, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        model.fit(1000)
    }
}