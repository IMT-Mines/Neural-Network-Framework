plugins {
    kotlin("jvm") version "1.9.23"
}

group = "org.neuralnetwork"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://packages.jetbrains.team/maven/p/kds/kotlin-ds-maven")
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlin-statistics-jvm:0.2.1")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}