package ru.itis.diploma.repositories;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.Vacancy;

import java.util.List;

public interface VacancyRepository extends JpaRepository<Vacancy, Long> {
    List<Vacancy> findAllByUserOrderByCreatedAtDesc(User user);
}
