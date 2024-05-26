package main.kotlin.train.deepReinforcement

import main.kotlin.network.*

class ReinforcementLearning {

    fun foodGameTraining() {
        val environment = FoodGameEnvironment()
        val model = NeuralNetwork(
            trainingMethod = QLearning(environment, 1000),
            optimizer = Adam(0.01, 0.9, 0.999),
            loss = MeanSquaredError
        )
        model.addLayer(Layer(25, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(40, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(20, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(FoodGameAction.entries.size, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        model.fit(2000)
        model.save("src/main/resources/foodGameModel.txt")
        model.test()
    }

    fun test() {
        val environment = FoodGameEnvironment()
        val model = NeuralNetwork(
            trainingMethod = QLearning(environment, 1000),
            optimizer = Adam(0.01, 0.9, 0.999),
            loss = MeanSquaredError
        )
        model.addLayer(Layer(25, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(40, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(20, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(FoodGameAction.entries.size, Softmax, NormalXavierGlorotInitialization))
        model.initialize()

        model.load("src/main/resources/foodGameModel.txt")
        model.test()
    }
}