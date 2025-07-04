package ru.itis.diploma.services.hr;

import jakarta.persistence.EntityNotFoundException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import ru.itis.diploma.models.User;
import ru.itis.diploma.models.hr.HrAttempt;
import ru.itis.diploma.repositories.hr.HrAttemptRepository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
@RequiredArgsConstructor
@Slf4j
public class HrAttemptService {

    private final HrAttemptRepository hrAttemptRepository;

    /**
     * Возвращает максимальный id попытки для пользователя.
     * Если у пользователя еще не было попыток, возвращает 1
     */
    public Long getNextAttemptIdForUser(User user) {
        return hrAttemptRepository.findMaxAttemptIdByUser(user) == null ? 1L :
                hrAttemptRepository.findMaxAttemptIdByUser(user) + 1;
    }

    /**
     * Создаёт новую попытку интервью для пользователя
     *
     * @param user Пользователь, для которого создаётся попытка
     * @return ID созданной попытки
     */
    public Long createNewAttempt(User user) {
        // Создаем новую попытку интервью
        HrAttempt attempt = HrAttempt.builder()
                .user(user)
                .startTime(LocalDateTime.now())
                .status(HrAttempt.AttemptStatus.IN_PROGRESS)
                .build();

        // Сохраняем попытку
        HrAttempt savedAttempt = hrAttemptRepository.save(attempt);

        return savedAttempt.getId();
    }

    /**
     * Получение всех попыток HR-интервью для определенного пользователя
     *
     * @return Список попыток HR-интервью
     */
    public List<HrAttempt> getUserAttempts(User user) {
        return hrAttemptRepository.findAllByUserOrderByEndTimeDesc(user);
    }

    /**
     * Получение конкретной попытки HR-интервью по ID
     *
     * @param attemptId ID попытки
     * @return Детали попытки HR-интервью
     * @throws EntityNotFoundException если попытка не найдена
     */
    public HrAttempt getAttemptById(Long attemptId) throws EntityNotFoundException {
        HrAttempt hrAttempt = hrAttemptRepository.findById(attemptId)
                .orElseThrow(() -> new EntityNotFoundException("Попытка HR-интервью с ID " + attemptId + " не найдена"));
        log.info("Извлечена попытка №" + attemptId + " "+ hrAttempt.getFeedback());
        return hrAttempt;
    }

    /**
     * Обновление попытки HR-интервью по ID
     *
     * @param attemptId ID попытки для обновления
     * @param updatedAttempt обновленные данные попытки
     * @return Обновленная попытка
     * @throws EntityNotFoundException если попытка с указанным ID не найдена
     */
    public HrAttempt updateAttempt(Long attemptId, HrAttempt updatedAttempt) {
        HrAttempt existingAttempt = hrAttemptRepository.findById(attemptId)
                .orElseThrow(() -> new EntityNotFoundException("Попытка HR-интервью с ID " + attemptId + " не найдена"));

        // Обновляем только не-null поля из updatedAttempt
        if (updatedAttempt.getStatus() != null) {
            existingAttempt.setStatus(updatedAttempt.getStatus());
        }
        if (updatedAttempt.getStartTime() != null) {
            existingAttempt.setStartTime(updatedAttempt.getStartTime());
        }
        if (updatedAttempt.getEndTime() != null) {
            existingAttempt.setEndTime(updatedAttempt.getEndTime());
        }
        if (updatedAttempt.getTotalScore() != null) {
            existingAttempt.setTotalScore(updatedAttempt.getTotalScore());
        }
        if (updatedAttempt.getFeedback() != null) {
            existingAttempt.setFeedback(updatedAttempt.getFeedback());
        }
        if (updatedAttempt.getVideoImageData() != null) {
            existingAttempt.setVideoImageData(updatedAttempt.getVideoImageData());
        }
        if (updatedAttempt.getAudioImageData() != null) {
            existingAttempt.setAudioImageData(updatedAttempt.getAudioImageData());
        }

        return hrAttemptRepository.save(existingAttempt);
    }


}
