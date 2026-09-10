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

tasks.register<Copy>("prepareNotebook") {
    group = "application"
    description = "Builds a stable local JAR for the Kotlin Notebook example."
    dependsOn(tasks.jar)
    from(tasks.jar)
    into(layout.buildDirectory.dir("notebook"))
    rename { "functional-ml-kotlin.jar" }
}

tasks.register<Exec>("notebook") {
    group = "application"
    description = "Builds the library and opens the example in JupyterLab."
    dependsOn("prepareNotebook")
    commandLine("jupyter", "lab", "notebooks/forward-mode-ad.ipynb")
}

tasks.test {
    useJUnitPlatform()
}
