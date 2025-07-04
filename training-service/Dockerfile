FROM openjdk:23-jdk-slim

WORKDIR /training-service

COPY target/*.jar training-service.jar

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "training-service.jar", "--spring.profiles.active=docker"]
