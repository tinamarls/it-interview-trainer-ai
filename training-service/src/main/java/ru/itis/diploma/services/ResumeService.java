package ru.itis.diploma.services;

import jakarta.annotation.Resource;
import lombok.RequiredArgsConstructor;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;
import ru.itis.diploma.repositories.ResumeRepository;

import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.nio.file.Paths;

@Service
@RequiredArgsConstructor
public class ResumeService {

    private final ResumeRepository resumeRepository;

    @Value("${file.upload-dir}")
    private String uploadDir;

    public List<Resume> getResumesByUser(User user) {
        return resumeRepository.findAllByUserOrderByUploadedAtDesc(user);
    }

    public Resume uploadResume(MultipartFile file, User user) {
        try {
            Files.createDirectories(Path.of(uploadDir));
            // Формируем путь к файлу
            String filePath = Paths.get(uploadDir, file.getOriginalFilename().replace(" ", "").replace(",", "_")).toString();
            java.nio.file.Path path = java.nio.file.Paths.get(filePath);

            // Извлекаем текст из PDF
            PDDocument document = PDDocument.load(file.getInputStream());
            if (document.getNumberOfPages() == 0) {
                throw new RuntimeException("PDF документ пуст.");
            }
            PDFTextStripper pdfStripper = new PDFTextStripper();
            String extractedText = pdfStripper.getText(document);
            document.close();

            // Сохраняем файл
            file.transferTo(path.toFile());

            if (extractedText == null || extractedText.trim().isEmpty()) {
                throw new RuntimeException("Не удалось извлечь текст из файла.");
            }

            // Создаём объект Resume
            Resume resume = Resume.builder()
                    .user(user)
                    .fileName(file.getOriginalFilename())
                    .filePath(filePath)
                    .extractedText(extractedText)
                    .uploadedAt(LocalDateTime.now())
                    .build();

            return resumeRepository.save(resume);
        } catch (IOException e) {
            throw new RuntimeException("Ошибка обработки файла: " + e.getMessage(), e);
        }
    }

    public Resume getResumeById(UUID id) {
        return resumeRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Резюме с id " + id + " не найдено"));
    }

}
