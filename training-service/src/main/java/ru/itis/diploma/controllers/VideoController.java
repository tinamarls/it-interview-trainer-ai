//package ru.itis.diploma.controllers;
//
//import lombok.RequiredArgsConstructor;
//import org.springframework.http.MediaType;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.PostMapping;
//import org.springframework.web.bind.annotation.RequestMapping;
//import org.springframework.web.bind.annotation.RequestParam;
//import org.springframework.web.bind.annotation.RestController;
//import org.springframework.web.multipart.MultipartFile;
//import reactor.core.publisher.Mono;
//import ru.itis.diploma.services.PythonService;
//
//import java.io.File;
//import java.io.FileOutputStream;
//
//@RestController
//@RequestMapping("/video")
//@RequiredArgsConstructor
//public class VideoController {
//
//    private final PythonService pythonAnalysisService;
//
//    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
//    public Mono<ResponseEntity<String>> uploadVideo(
//            @RequestParam("video") MultipartFile video,
//            @RequestParam("questionId") Long questionId) {
//
//        return saveToTempFile(video)
//                .flatMap(pythonAnalysisService::analyzeVideo)
//                .map(result -> ResponseEntity.ok("Результат анализа: " + result))
//                .onErrorResume(error -> {
//                    return Mono.just(ResponseEntity.internalServerError()
//                            .body("Ошибка при анализе видео: " + error.getMessage()));
//                });
//    }
//
//    private Mono<File> saveToTempFile(MultipartFile multipartFile) {
//        return Mono.fromCallable(() -> {
//            File tempFile = File.createTempFile("uploaded-", ".webm");
//            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
//                fos.write(multipartFile.getBytes());
//            }
//            return tempFile;
//        });
//    }
//}
//
