package ru.itis.diploma.controllers;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;
import ru.itis.diploma.dto.response.ComparasionResult;
import ru.itis.diploma.dto.response.VacancyDto;
import ru.itis.diploma.models.ResumeVacancyMatch;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.Vacancy;
import ru.itis.diploma.repositories.ResumeRepository;
import ru.itis.diploma.repositories.ResumeVacancyMatchRepository;
import ru.itis.diploma.repositories.VacancyRepository;
import ru.itis.diploma.services.ResumeService;
import ru.itis.diploma.services.UserService;
import ru.itis.diploma.services.VacancyService;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

//@RestController
//@RequestMapping("/vacancy")
//@RequiredArgsConstructor
//public class VacancyController {
//
//    private final VacancyService vacancyService;
//
//    @PostMapping("/analyze")
//    public ResponseEntity<Map<String, String>> analyzeVacancy(@RequestBody Map<String, String> request) {
//        String text = request.get("text");
//        Vacancy vacancy = vacancyService.analyzeAndSaveVacancy(text);
//        return ResponseEntity.ok(Map.of("text", vacancy.getExtractedText()));
//    }
//}

@RestController
@RequestMapping("/vacancy")
@RequiredArgsConstructor
public class VacancyController {

    private final VacancyService vacancyService;
    private final UserService userService;
    private final ResumeVacancyMatchRepository resumeVacancyMatchRepository;
    private final VacancyRepository vacancyRepository;
    private final ResumeService resumeService;

    @PostMapping("/save")
    public ResponseEntity<Vacancy> saveVacancy(@RequestBody Map<String, String> request,
                                               @AuthenticationPrincipal UserDetails userDetails) {
        User user = userService.getUserByEmail(userDetails.getUsername());
        String title = request.get("title");
        String url = request.get("url");
        Vacancy vacancy = vacancyService.saveVacancy(url, title, user);
        return ResponseEntity.ok(vacancy);
    }

    @GetMapping("/all")
    public ResponseEntity<List<VacancyDto>> getAllVacancies(@AuthenticationPrincipal UserDetails userDetails) {
        User user = userService.getUserByEmail(userDetails.getUsername());
        List<Vacancy> vacancies = vacancyService.getAllVacanciesByUser(user);

        List<VacancyDto> response = vacancies.stream()
                .map(v -> new VacancyDto(v.getId(), v.getTitle(), v.getUrl(), v.getCreatedAt(), v.getExtractedText()))
                .collect(Collectors.toList());

        return ResponseEntity.ok(response);
    }

    @GetMapping("/compare")
    public ResponseEntity<ComparasionResult> compareResumeWithVacancy(
            @RequestParam("resume_id") String resumeId,
            @RequestParam("vacancy_id") String vacancyId,
            @AuthenticationPrincipal UserDetails userDetails) {
        User user = userService.getUserByEmail(userDetails.getUsername());
        ComparasionResult result = vacancyService.compareResumeWithVacancy(resumeId, vacancyId, user);
        if (result != null) {
            resumeVacancyMatchRepository.save(
                    ResumeVacancyMatch.builder()
                            .resume(resumeService.getResumeById(UUID.fromString(resumeId)))
                            .vacancy(vacancyRepository.findById(Long.valueOf(vacancyId)).orElse(null))
                            .matchScore(result.getMatchingPercent())
                            .recommendations(result.getNeededSkills())
                            .user(user)
                            .matchTimestamp(LocalDateTime.now())
                            .build()
            );
        }

        return ResponseEntity.ok(result);
    }

}

