//package ru.itis.diploma.services;
//
//import com.fasterxml.jackson.databind.JsonNode;
//import com.fasterxml.jackson.databind.ObjectMapper;
//import lombok.RequiredArgsConstructor;
//import lombok.extern.slf4j.Slf4j;
//import org.springframework.core.ParameterizedTypeReference;
//import org.springframework.core.io.FileSystemResource;
//import org.springframework.http.MediaType;
//import org.springframework.stereotype.Service;
//import org.springframework.web.reactive.function.BodyInserters;
//import org.springframework.web.reactive.function.client.WebClient;
//import reactor.core.publisher.Mono;
//import ru.itis.diploma.dto.response.VacancyResponse;
//
//import java.io.File;
//import java.time.Duration;
//import java.util.Map;
//import java.util.Objects;
//
//@Service
//@RequiredArgsConstructor
//@Slf4j
//public class PythonService {
//
//    private final WebClient webClient = WebClient.builder()
//            .baseUrl("http://host.docker.internal:5000")
//            .build();
//
//    public Mono<String> analyzeVideo(File videoFile) {
//        return webClient.post()
//                .uri("/video/analyze")
//                .contentType(MediaType.MULTIPART_FORM_DATA)
//                .body(BodyInserters.fromMultipartData("video", new FileSystemResource(videoFile)))
//                .retrieve()
//                .bodyToMono(String.class)
//                .doFinally(signal -> {
//                    // удалим временный файл после завершения запроса
//                    if (videoFile.exists()) {
//                        videoFile.delete();
//                    }
//                });
//    }
//
/// /    public String getVacancyText(String url) {
/// /        return Objects.requireNonNull(webClient.get()
/// /                .uri(uriBuilder -> uriBuilder
/// /                        .path("/vacancy/parse_vacancy")
/// /                        .queryParam("url", url)
/// /                        .build())
/// /                .retrieve()
/// /                .bodyToMono(VacancyResponse.class)
/// /                .block())
/// /                .extracted_text;
/// /    }
//
/// /    public String getVacancyText(String url) {
/// /        String response = webClient.get()
/// /                .uri(uriBuilder -> uriBuilder
/// /                        .path("/vacancy/parse_vacancy")
/// /                        .queryParam("url", url)
/// /                        .build())
/// /                .retrieve()
/// /                .bodyToMono(String.class)
/// /                .block();
/// /
/// /        try {
/// /            ObjectMapper mapper = new ObjectMapper();
/// /            JsonNode node = mapper.readTree(response);
/// /            System.out.printf(node.get("extracted_text").asText());
/// /            return node.get("extracted_text").asText();
/// /        } catch (Exception e) {
/// /            e.printStackTrace();
/// /            return "";
/// /        }
/// /    }
//
//    public String getVacancyText(String url) {
//        VacancyResponse response = webClient.get()
//                .uri(uriBuilder -> uriBuilder
//                        .path("/vacancy/parse_vacancy")
//                        .queryParam("url", url)
//                        .build())
//                .retrieve()
//                .bodyToMono(VacancyResponse.class)
//                .block();
//
//        log.info("Extracted text from Python service: {}", response.extracted_text);
//
//        return response != null ? response.extracted_text : "";
//    }
//
//    /**
//     * Отправляет два текста на Python-сервис для сравнения и анализа
//     *
//     * @param resumeText  текст резюме
//     * @param vacancyText текст вакансии
//     * @return результат сравнения в виде Map с данными
//     */
//    public Map<String, Object> compareTexts(String resumeText, String vacancyText) {
//        log.info("Отправка запроса на сравнение текстов в Python-сервис");
//
//        // Создаем тело запроса с двумя текстами
//        Map<String, String> requestBody = Map.of(
//                "resume_text", resumeText,
//                "vacancy_text", vacancyText
//        );
//
//        try {
//            // Отправляем POST запрос на эндпоинт сравнения текстов
//            Map<String, Object> response = webClient.post()
//                    .uri("/vacancy/compare")
//                    .contentType(MediaType.APPLICATION_JSON)
//                    .bodyValue(requestBody)
//                    .retrieve()
//                    .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {
//                    })
//                    .timeout(Duration.ofSeconds(30))
//                    .block();
//
//            log.info("Получен ответ от Python-сервиса: {}", response);
//
//            return response != null ? response : Map.of();
//        } catch (Exception e) {
//            log.error("Ошибка при обращении к Python-сервису для сравнения текстов", e);
//            // Возвращаем пустую карту в случае ошибки
//            return Map.of();
//        }
//    }
//}
//
//


package ru.itis.diploma.services;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import ru.itis.diploma.dto.response.*;
import ru.itis.diploma.repositories.hr.HrAttemptRepository;
import ru.itis.diploma.services.hr.HrAttemptService;

import java.io.File;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

@Service
@RequiredArgsConstructor
@Slf4j
public class PythonService {

//    private final WebClient webClient = WebClient.builder()
//            .baseUrl("http://host.docker.internal:5000")
//            .build();

    private final WebClient webClient = WebClient.builder()
            .baseUrl("http://localhost:5000")
            .build();

    private final ObjectMapper objectMapper;

    private final HrAttemptService hrAttemptService;

//    /**
//     * Оригинальный метод для анализа видео, возвращающий сырую JSON-строку
//     */
//    public Mono<String> analyzeVideo(File videoFile) {
//        return webClient.post()
//                .uri("/analyze") // URI соответствует Python FastAPI
//                .contentType(MediaType.MULTIPART_FORM_DATA)
//                .body(BodyInserters.fromMultipartData("video", new FileSystemResource(videoFile)))
//                .retrieve()
//                .bodyToMono(String.class)
//                .doFinally(signal -> {
//                    // удалим временный файл после завершения запроса
//                    if (videoFile.exists()) {
//                        if (videoFile.delete()) {
//                            log.debug("Временный видеофайл успешно удален: {}", videoFile.getAbsolutePath());
//                        } else {
//                            log.warn("Не удалось удалить временный видеофайл: {}", videoFile.getAbsolutePath());
//                        }
//                    }
//                });
//    }

    /**
     * Метод для анализа видео, возвращающий структурированный объект VideoAnalysisResponse
     */
    public Mono<VideoAnalysisResponse> analyzeVideoAndGetResponse(File videoFile) {
        log.info("Отправка видеофайла {} на анализ", videoFile.getName());

        return webClient.post()
                .uri("video/analyze_my_new") // URI соответствует Python FastAPI
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("video", new FileSystemResource(videoFile)))
                .retrieve()
                .bodyToMono(String.class)
                .map(jsonResponse -> {
                    try {
                        log.info("Получен ответ от Python-сервиса для видео {}", videoFile.getName());
//                         Преобразуем JSON-строку в VideoAnalysisResponse
                        System.out.println(jsonResponse);
                        return parseVideoAnalysisResponse(jsonResponse);
                    } catch (Exception e) {
                        log.error("Ошибка при обработке ответа Python-сервиса: {}", e.getMessage(), e);
                        throw new RuntimeException("Ошибка при обработке ответа от сервиса анализа видео", e);
                    }
                })
                .doFinally(signal -> {
                    // Удаляем временный файл после завершения запроса
//                    if (videoFile.exists()) {
//                        if (videoFile.delete()) {
//                            log.info("Временный видеофайл {} успешно удален", videoFile.getAbsolutePath());
//                        } else {
//                            log.warn("Не удалось удалить временный видеофайл: {}", videoFile.getAbsolutePath());
//                        }
//                    }
                });
    }

    /**
     * Парсит JSON-ответ от Python-сервиса в объект VideoAnalysisResponse
     */
    private VideoAnalysisResponse parseVideoAnalysisResponse(String jsonResponse) {
        try {
            JsonNode rootNode = objectMapper.readTree(jsonResponse);
            System.out.println(jsonResponse);

            // Проверяем на наличие ошибки верхнего уровня
            if (rootNode.has("error")) {
                VideoAnalysisResponse response = new VideoAnalysisResponse();
                response.setError(rootNode.get("error").asText());
                return response;
            }

            VideoAnalysisResponse response = new VideoAnalysisResponse();
            response.setResult(jsonResponse);
            response.setVideoImagePath("/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/" + rootNode.get("plot_path").asText());
            response.setAudioImagePath("/Users/kristina/IdeaProjects/it-train-diploma/machine-learning-service/" + rootNode.get("audio_plot_path").asText());
            return response;
        } catch (Exception e) {
            log.error("Ошибка при парсинге JSON-ответа: {}", e.getMessage(), e);
            throw new RuntimeException("Ошибка при парсинге ответа от сервиса анализа видео", e);
        }
    }

    private ComparasionResult parseComparationResponse(String jsonResponse) {
        try {
            JsonNode rootNode = objectMapper.readTree(jsonResponse);
            if (rootNode.has("error")) {
                ComparasionResult response = new ComparasionResult();
                response.setError(rootNode.get("error").asText());
                return response;
            }

            ComparasionResult response = new ComparasionResult();
            response.setFullResultJson(jsonResponse);
            response.setMatchingPercent(rootNode.get("matching_percent").asDouble());
            response.setNeededSkills(rootNode.get("needed_skills").asText());
            return response;
        } catch (Exception e) {
            log.error("Ошибка при парсинге JSON-ответа: {}", e.getMessage(), e);
            throw new RuntimeException("Ошибка при парсинге ответа от сервиса сравнения", e);
        }
    }

    /**
     * Получение текста вакансии по URL
     */
    public String getVacancyText(String url) {
        VacancyResponse response = webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/vacancy/parse_vacancy")
                        .queryParam("url", url)
                        .build())
                .retrieve()
                .bodyToMono(VacancyResponse.class)
                .block();

        log.info("Extracted text from Python service: {}", response != null ? response.extracted_text : "null");

        return response != null ? response.extracted_text : "";
    }

    /**
     * Сравнение текстов резюме и вакансии
     */
    public Mono<ComparasionResult> compareTexts(String resumeText, String vacancyText) {
        log.info("Отправка запроса на сравнение текстов в Python-сервис");

        // Создаем тело запроса с двумя текстами
        Map<String, String> requestBody = Map.of(
                "resume_text", resumeText,
                "vacancy_text", vacancyText
        );

        try {
            // Отправляем POST запрос на эндпоинт сравнения текстов
            return webClient.post()
                    .uri("/vacancy/compare")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(String.class)
                    .timeout(Duration.ofSeconds(30))
                    .map(jsonResponse -> {
                        try {
                            log.info("Получен ответ от Python-сервиса для сравнения вакансий");
                            System.out.println(jsonResponse);
                            return parseComparationResponse(jsonResponse);
                        } catch (Exception e) {
                            log.error("Ошибка при обработке ответа Python-сервиса: {}", e.getMessage(), e);
                            throw new RuntimeException("Ошибка при обработке ответа от сервиса сравнения", e);
                        }
                    });
        } catch (Exception e) {
            throw new RuntimeException("Ошибка при обработке ответа от сервиса сравнения", e);
        }
    }
}