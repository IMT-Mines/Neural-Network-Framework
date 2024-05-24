plugins {
    kotlin("jvm") version "1.9.23"
    `maven-publish`
}

group = "org.neuralnetwork"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://packages.jetbrains.team/maven/p/kds/kotlin-ds-maven")
}

dependencies {
    implementation(kotlin("reflect"))
    implementation("org.jetbrains.kotlinx:kotlin-statistics-jvm:0.2.1")
    implementation("org.slf4j:slf4j-simple:2.0.13")
    implementation("org.slf4j:slf4j-api:2.0.13")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("mavenJava") {
                from(components["java"])
                groupId = project.group.toString()
                artifactId = project.name
                version = project.version.toString()
            }
        }
    }
}