package ru.itis.diploma.controllers;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.hr.HrAttempt;
import ru.itis.diploma.services.UserService;
import ru.itis.diploma.services.hr.HrAttemptService;
import ru.itis.diploma.services.hr.HrResponseService;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/hr-interview")
@RequiredArgsConstructor
public class HrScreeningController {

    private final HrAttemptService hrAttemptService;
    private final UserService userService;
    private final HrResponseService hrResponseService;

    /**
     * Эндпоинт для начала нового HR-интервью для пользователя
     *
     * @param userDetails Информация о текущем аутентифицированном пользователе
     * @return ID созданной попытки интервью
     */
    @PostMapping("")
    public ResponseEntity<?> startHrInterview(@AuthenticationPrincipal UserDetails userDetails) {
        try {
            // Получаем пользователя из UserDetails
            User user = userService.getUserByEmail(userDetails.getUsername());

            // Создаем новую попытку интервью через сервис
            Long attemptId = hrAttemptService.createNewAttempt(user);

            // Возвращаем только ID
            return ResponseEntity.ok(Map.of("id", attemptId));
        } catch (EntityNotFoundException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Произошла ошибка при создании HR-интервью: " + e.getMessage()));
        }
    }

    @PostMapping(value = "/upload-video", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<ResponseEntity<String>> uploadVideo(
            @RequestParam("video") MultipartFile video,
            @RequestParam("attemptId") Long attemptId) {

        return hrResponseService.processVideoResponse(video, attemptId)
                .map(result -> ResponseEntity.ok("Результат анализа: " + result))
                .onErrorResume(error -> {
                    return Mono.just(ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body("Ошибка при обработке видео: " + error.getMessage()));
                });
    }

    /**
     * Получение списка всех попыток HR-интервью для текущего пользователя
     *
     * @param userDetails Информация о текущем аутентифицированном пользователе
     * @return Список попыток HR-интервью пользователя
     */
    @GetMapping("/results")
    public ResponseEntity<?> getAllUserAttempts(@AuthenticationPrincipal UserDetails userDetails) {
        try {
            User user = userService.getUserByEmail(userDetails.getUsername());
            List<HrAttempt> attempts = hrAttemptService.getUserAttempts(user);

            return ResponseEntity.ok(attempts);
        } catch (EntityNotFoundException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Произошла ошибка при получении списка попыток: " + e.getMessage()));
        }
    }

    /**
     * Получение деталей конкретной попытки HR-интервью по ID
     *
     * @param attemptId   ID попытки
     * @param userDetails Информация о текущем аутентифицированном пользователе
     * @return Детали попытки HR-интервью
     */
    @GetMapping("/results/{attemptId}")
    public ResponseEntity<?> getAttemptById(
            @PathVariable Long attemptId,
            @AuthenticationPrincipal UserDetails userDetails) {
        try {
            HrAttempt attempt = hrAttemptService.getAttemptById(attemptId);

            return ResponseEntity.ok(attempt);
        } catch (EntityNotFoundException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Произошла ошибка при получении деталей попытки: " + e.getMessage()));
        }
    }


}
