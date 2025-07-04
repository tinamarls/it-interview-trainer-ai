package ru.itis.diploma.repositories.hr;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import ru.itis.diploma.models.Resume;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.hr.HrAttempt;

import java.util.List;

public interface HrAttemptRepository extends JpaRepository<HrAttempt, Long> {
    /**
     * Находит максимальный идентификатор попытки для указанного пользователя.
     * Если попыток нет, запрос вернет null
     */
    @Query("SELECT MAX(a.id) FROM HrAttempt a WHERE a.user = :user")
    Long findMaxAttemptIdByUser(@Param("user") User user);

    List<HrAttempt> findAllByUserOrderByEndTimeDesc(User user);
}
