package ru.itis.diploma.controllers;

import org.springframework.core.io.Resource;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;
import ru.itis.diploma.dto.request.ResumeDto;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;
import ru.itis.diploma.services.ResumeService;
import ru.itis.diploma.services.UserService;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/resume")
@RequiredArgsConstructor
public class ResumeController {

    private final ResumeService resumeService;
    private final UserService userService;

    @PostMapping("/upload")
    public ResponseEntity<String> uploadResume(@RequestParam("file") MultipartFile file,
                                               @AuthenticationPrincipal UserDetails userDetails) {
        User user = userService.getUserByEmail(userDetails.getUsername());
        Resume resume = resumeService.uploadResume(file, user);
        return ResponseEntity.ok("Резюме загружено, текст сохранен в БД");
    }

    @GetMapping("/download/{id}")
    public ResponseEntity<Resource> downloadResume(@PathVariable UUID id) {
        Resume resume = resumeService.getResumeById(id); // Метод для получения резюме по id
        Path filePath = Path.of(resume.getFilePath());

        if (!Files.exists(filePath)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Файл не найден");
        }

        Resource resource = new FileSystemResource(filePath.toFile());

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resume.getFileName() + "\"")
                .body(resource);
    }

    @GetMapping("/all")
    public ResponseEntity<List<ResumeDto>> getAllResumes(@AuthenticationPrincipal UserDetails userDetails) {
        User user = userService.getUserByEmail(userDetails.getUsername());
        List<Resume> resumes = resumeService.getResumesByUser(user);

        List<ResumeDto> response = resumes.stream()
                .map(r -> new ResumeDto(r.getId(), r.getFileName(), r.getUploadedAt()))
                .collect(Collectors.toList());

        return ResponseEntity.ok(response);
    }

}

