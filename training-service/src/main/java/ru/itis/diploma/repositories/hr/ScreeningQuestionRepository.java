package ru.itis.diploma.repositories.hr;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.hr.ScreeningQuestion;

public interface ScreeningQuestionRepository extends JpaRepository<ScreeningQuestion, Long> {
}
