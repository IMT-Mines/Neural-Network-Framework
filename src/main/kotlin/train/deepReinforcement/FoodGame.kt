package main.kotlin.train.deepReinforcement

import main.kotlin.network.*
import main.kotlin.reinforcement.Action
import main.kotlin.reinforcement.PPOAgent

class FoodGame {


    fun foodGame() {
        val environment = FoodGameEnvironment()
        val model = NeuralNetwork(optimizer = Adam(0.01, 0.9, 0.999))
        model.addLayer(Layer(Action.entries.size, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(128, LeakyReLU, NormalHeInitialization))
        model.addLayer(Layer(128, LeakyReLU, NormalHeInitialization))


        val agent = PPOAgent(model, environment)


        agent.train(1000, 1, 0.99, 0.2)
    }

}