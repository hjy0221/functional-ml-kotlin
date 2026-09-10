plugins {
    kotlin("jvm") version "2.2.20"
    application
}

group = "dev.functionalml"
version = "0.1.0-SNAPSHOT"

kotlin {
    jvmToolchain(21)
}

dependencies {
    testImplementation(kotlin("test"))
}

application {
    mainClass = "dev.functionalml.MainKt"
}

tasks.test {
    useJUnitPlatform()
}
