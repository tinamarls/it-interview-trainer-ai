package ru.itis.diploma.repositories.hr;

import org.springframework.data.jpa.repository.JpaRepository;
import ru.itis.diploma.models.hr.HrResponse;

public interface HrResponseRepository extends JpaRepository<HrResponse, Long> {
}
