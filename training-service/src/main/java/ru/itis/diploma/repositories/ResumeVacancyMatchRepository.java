package ru.itis.diploma.repositories;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.ResumeVacancyMatch;

public interface ResumeVacancyMatchRepository extends JpaRepository<ResumeVacancyMatch, Long> {
}
