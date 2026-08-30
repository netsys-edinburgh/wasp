import com.google.protobuf.gradle.*

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.jetbrainsKotlinAndroid)
    id("com.google.protobuf") version "0.9.5"
}

android {
    namespace = "com.trainingontheedge.androidwaspclient"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.trainingontheedge.androidwaspclient"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    protobuf {
        protoc {
            // Use the desired version of protoc
            artifact = "com.google.protobuf:protoc:4.30.2"
        }
        generateProtoTasks {
            all().forEach { task ->
                task.builtins {
                    // Generate Java stubs; these work well with Kotlin.
                    id("java")
                }
            }
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("build/generated/source/proto/main/java")
        }
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    implementation("com.google.ai.edge.litert:litert:+")
    implementation("com.google.ai.edge.litert:litert-api:+")
    implementation("com.google.ai.edge.litert:litert-gpu:+")
    implementation("com.google.ai.edge.litert:litert-gpu-api:+")
    implementation("com.google.ai.edge.litert:litert-support:+")
    implementation("com.google.ai.edge.litert:litert-support-api:+")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.15.2")
    implementation("com.google.protobuf:protobuf-java:4.30.2")
    implementation("com.google.protobuf:protobuf-kotlin:4.30.2")
    implementation("com.koushikdutta.async:androidasync:2.+")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}