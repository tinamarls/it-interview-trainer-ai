package ru.itis.diploma.services;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import ru.itis.diploma.dto.response.ComparasionResult;
import ru.itis.diploma.dto.response.ComparisonResultDto;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.Vacancy;
import ru.itis.diploma.repositories.ResumeRepository;
import ru.itis.diploma.repositories.VacancyRepository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@Service
@RequiredArgsConstructor
public class VacancyService {

    private final PythonService pythonService;
    private final VacancyRepository vacancyRepository;
    private final ResumeRepository resumeRepository;

    public Vacancy saveVacancy(String url, String title, User user) {
        String extractedText = pythonService.getVacancyText(url);

        Vacancy vacancy = Vacancy.builder()
                .url(url)
                .extractedText(extractedText)
                .createdAt(LocalDateTime.now())
                .title(title)
                .user(user)
                .build();

        return vacancyRepository.save(vacancy);
    }

    public List<Vacancy> getAllVacanciesByUser(User user) {
        return vacancyRepository.findAllByUserOrderByCreatedAtDesc(user);
    }

    public ComparasionResult compareResumeWithVacancy(String resumeId, String vacancyId, User user) {
        Resume resume = resumeRepository.findById(UUID.fromString(resumeId))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Резюме не найдено"));

        Vacancy vacancy = vacancyRepository.findById(Long.parseLong(vacancyId))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Вакансия не найдена"));

        String resumeText = resume.getExtractedText();
        String vacancyText = vacancy.getExtractedText();

        return pythonService.compareTexts(resumeText, vacancyText).block();
    }

}
