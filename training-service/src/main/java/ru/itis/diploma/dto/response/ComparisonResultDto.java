package ru.itis.diploma.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ComparisonResultDto {
    // Процент соответствия резюме вакансии
    private Double matchPercentage;

    // Можно добавить детальную информацию о соответствии по навыкам
    private Map<String, Double> skillsMatch;

    // Рекомендации по улучшению резюме
    private String recommendations;

    // Для отображения на фронте - исходные данные
    private String resumeText;
    private String vacancyText;
}
