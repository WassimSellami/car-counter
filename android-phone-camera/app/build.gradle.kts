import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.util.Properties

val senderEnv = Properties().apply {
    val envFile = rootProject.projectDir.parentFile.resolve(".env")
    if (envFile.isFile) envFile.inputStream().use(::load)
}

fun envValue(name: String): String = senderEnv.getProperty(name, "").trim().removeSurrounding("\"").removeSurrounding("'")
fun buildConfigString(value: String): String = "\"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\""

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.carcounter.phonecamera"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.carcounter.phonecamera"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        buildConfigField("String", "DEFAULT_COUNTER_URL", buildConfigString(envValue("COUNTER_PUBLIC_URL")))
        buildConfigField("String", "DEFAULT_API_KEY", buildConfigString(envValue("CLOUD_INFERENCE_API_KEY")))
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")
    implementation("androidx.lifecycle:lifecycle-service:2.8.7")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
