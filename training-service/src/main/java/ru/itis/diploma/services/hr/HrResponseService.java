package ru.itis.diploma.services.hr;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import ru.itis.diploma.dto.response.VideoAnalysisResponse;
import ru.itis.diploma.models.hr.HrAttempt;
import ru.itis.diploma.models.hr.HrResponse;
import ru.itis.diploma.models.hr.ScreeningQuestion;
import ru.itis.diploma.repositories.hr.HrAttemptRepository;
import ru.itis.diploma.repositories.hr.HrResponseRepository;
import ru.itis.diploma.repositories.hr.ScreeningQuestionRepository;
import ru.itis.diploma.services.PythonService;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
@Slf4j
public class HrResponseService {

    private final HrResponseRepository hrResponseRepository;
    private final HrAttemptRepository hrAttemptRepository;
    private final ScreeningQuestionRepository screeningQuestionRepository;
    private final PythonService pythonService;
    private final ObjectMapper objectMapper;
    private final HrAttemptService hrAttemptService;

    public Mono<String> processVideoResponse(MultipartFile video, Long attemptId) {
        log.info("Начало обработки видео-ответа, questionId: {}, attemptId: {}", attemptId);

        return Mono.fromCallable(() -> {
                    // Сохраняем запись об ответе
                    return saveResponseRecord(attemptId);
                })
                .flatMap(responseId -> saveToTempFile(video)
                        .flatMap(filePath -> {
                            log.info("Файл сохранен, начинаем анализ: {}", filePath.getName());
                            return pythonService.analyzeVideoAndGetResponse(filePath)
                                    .publishOn(Schedulers.boundedElastic())
                                    .map(videoAnalysisResponse -> {
                                        try {
                                            // Создаем объект с обновляемыми полями
                                            HrAttempt updates = HrAttempt.builder()
                                                    .status(HrAttempt.AttemptStatus.COMPLETED)  // Устанавливаем статус "Завершено"
                                                    .endTime(LocalDateTime.now())               // Текущее время как время завершения
                                                    .feedback(videoAnalysisResponse.getResult())  // Ваш фидбэк
                                                    .videoImageData(Files.readAllBytes(Path.of(videoAnalysisResponse.getVideoImagePath())))
                                                    .audioImageData(Files.readAllBytes(Path.of(videoAnalysisResponse.getAudioImagePath())))
                                                    .build();
                                            System.out.println("Обновление попытки: " + updates.getId() + " \n\n\n");

                                            hrAttemptService.updateAttempt(attemptId, updates);
                                            return videoAnalysisResponse.toString();
                                        } catch (Exception e) {
                                            log.error("Ошибка при сериализации: {}", e.getMessage(), e);
                                            return "Ошибка сериализации: " + e.getMessage();
                                        }
                                    })
                                    .timeout(Duration.ofMinutes(3))
                                    .doOnSuccess(result -> {
                                        log.info("Анализ успешно завершен для responseId: {}", responseId);
                                        // Асинхронно обновляем результаты анализа в БД
                                        // updateResponseWithAnalysisResult(responseId, result);
                                    })
                                    .doOnError(error -> {
                                        log.error("Ошибка при анализе видео: {}", error.getMessage(), error);
                                    });
                        })
                        .onErrorResume(e -> {
                            log.error("Ошибка при обработке видео: {}", e.getMessage(), e);
                            return Mono.just("Ошибка при обработке видео: " + e.getMessage());
                        })
                )
                .subscribeOn(Schedulers.boundedElastic());
            }

    public Long saveResponseRecord(Long attemptId) {
        try {
            // Находим попытку по ID
            HrAttempt attempt = hrAttemptRepository.findById(attemptId)
                    .orElseThrow(() -> new EntityNotFoundException("Попытка с ID " + attemptId + " не найдена"));

            // Создаем и сохраняем запись об ответе
            HrResponse response = HrResponse.builder()
                    .interviewAttempt(attempt)
                    .responseTime(LocalDateTime.now())
                    .build();

            HrResponse savedResponse = hrResponseRepository.save(response);
            return savedResponse.getId();
        } catch (Exception e) {
            log.error("Ошибка при сохранении записи ответа: {}", e.getMessage(), e);
            throw e;
        }
    }

    private Mono<File> saveToTempFile(MultipartFile multipartFile) {
        return Mono.fromCallable(() -> {
            File tempFile = File.createTempFile("uploaded_new-", ".webm");
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                fos.write(multipartFile.getBytes());
            }
            return tempFile;
        });
    }
}
